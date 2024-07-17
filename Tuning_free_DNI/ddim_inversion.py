from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torchvision.io import write_video, read_video
from pathlib import Path
from tqdm import tqdm, trange
from PIL import Image
from dni_utils import *
import os
import yaml
import random
import argparse
import shutil
import pdb

def seed_fix(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start

@torch.no_grad()
def encode_imgs(vae, imgs, deterministic=True):
    imgs = 2 * imgs - 1
    latents = []
    batch_size = 8
    for i in range(0, len(imgs), batch_size):
        posterior = vae.encode(imgs[i:i + batch_size]).latent_dist
        latent = posterior.mean if deterministic else posterior.sample()
        latents.append(latent * 0.18215)
    latents = torch.cat(latents)
    return latents

@torch.no_grad()
def decode_latents(vae, latents):
    decoded = []
    batch_size = 8
    for b in range(0, latents.shape[0], batch_size):
            latents_batch = 1 / 0.18215 * latents[b:b + batch_size]
            imgs = vae.decode(latents_batch).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
            decoded.append(imgs)
    return torch.cat(decoded)

@torch.no_grad()
def get_text_embeds(tokenizer, text_encoder, prompt, negative_prompt, device):
    text_input = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors='pt')
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    uncond_input = tokenizer(negative_prompt, padding='max_length', max_length=tokenizer.model_max_length,
                                  return_tensors='pt')
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings

@torch.no_grad()
def ddim_inversion(cond, latent, scheduler, unet, save_path, save_steps, batch_size=100):
    timesteps = reversed(scheduler.timesteps)
    for i, t in enumerate(tqdm(timesteps)):
        for b in range(0, latent.shape[0], batch_size):
            latent_batch = latent[b:b + batch_size]
            model_input = latent_batch
            cond_batch = cond.repeat(latent_batch.shape[0], 1, 1)

            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[timesteps[i - 1]]
                if i > 0 else scheduler.final_alpha_cumprod
            )

            mu = alpha_prod_t ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

            eps = unet(model_input, t, encoder_hidden_states=cond_batch).sample
            pred_x0 = (latent_batch - sigma_prev * eps) / mu_prev
            latent[b:b + batch_size] = mu * pred_x0 + sigma * eps
        if t in save_steps:
            torch.save(latent, os.path.join(save_path, f'ddim_latents_{t}.pt'))
    torch.save(latent, os.path.join(save_path, f'ddim_latents_{t}.pt'))
    return latent

@torch.no_grad()
def ddim_sample_with_target(cond, latent, scheduler, unet, opt, tgt_cond, batch_size=100):
    timesteps = scheduler.timesteps
    register_target_save(unet, False)
    for i, t in enumerate(tqdm(timesteps)):
        for b in range(0, latent.shape[0], batch_size):
            latent_batch = latent[b:b + batch_size]
            model_input = latent_batch
            cond_batch = cond.repeat(latent_batch.shape[0], 1, 1)
            
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else scheduler.final_alpha_cumprod
            )
            mu = alpha_prod_t ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

            eps = unet(model_input, t, encoder_hidden_states=cond_batch).sample

            pred_x0 = (latent_batch - sigma * eps) / mu
            latent[b:b + batch_size] = mu_prev * pred_x0 + sigma_prev * eps

        if t == opt.target_step:
            tgt_cond = tgt_cond.repeat(latent.shape[0], 1, 1)
            model_input = latent
            register_target_save(unet, True)
            edit_guide = unet(model_input, t, encoder_hidden_states=tgt_cond)
            register_target_save(unet, False)
    return latent

def run(opt):

    seed_fix(opt.seed)

    # out dir setting
    if os.path.isdir(os.path.join(opt.out_dir, opt.video_name)):
        shutil.rmtree(os.path.join(opt.out_dir, opt.video_name))
    save_path = os.path.join(opt.out_dir, opt.video_name, 'latents')
    save_prompt_path = os.path.join(opt.out_dir, opt.video_name, 'source_prompt')
    save_recon_video_path = os.path.join(opt.out_dir, opt.video_name, 'recon_video')
    save_edit_guidance_path = os.path.join(opt.out_dir, opt.video_name, 'edit_guidance')

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_prompt_path, exist_ok=True)
    os.makedirs(save_recon_video_path, exist_ok=True)
    os.makedirs(save_edit_guidance_path, exist_ok=True)

    with open(os.path.join(save_prompt_path, 'source_prompt.txt'), 'w') as f:
        f.write(opt.source_prompt)

    ######### Stable Diffusion Setting ###################
    # sd version, empirically 2.1 gives more better results
    sd_v = "stabilityai/stable-diffusion-2-1-base"
    ddim_scheduler = DDIMScheduler.from_pretrained(sd_v, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(sd_v, subfolder="vae", revision="fp16", torch_dtype=torch.float16).to(opt.device)
    tokenizer = CLIPTokenizer.from_pretrained(sd_v, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_v, subfolder="text_encoder", revision="fp16", torch_dtype=torch.float16).to(opt.device)
    unet = UNet2DConditionModel.from_pretrained(sd_v, subfolder="unet", revision="fp16", torch_dtype=torch.float16).to(opt.device)

    ######### input video latent #########################
    video,_,meta = read_video(opt.input_video, output_format="TCHW")
    fps = meta['video_fps']
    frames = []
    for i in range(len(video)):
        image = T.ToPILImage()(video[i])
        image = image.resize((opt.h, opt.w),  resample=Image.Resampling.LANCZOS)
        frame = image.convert('RGB')
        frames = frames + [frame]
    frames = torch.stack([T.ToTensor()(frame) for frame in frames]).to(torch.float16).to(opt.device)
    latents = encode_imgs(vae, frames, deterministic=True).to(torch.float16).to(opt.device)

    ######### DDIM latent ################################
    # set timesteps for ddim latent save point
    ddim_scheduler.set_timesteps(opt.save_steps)
    save_steps = ddim_scheduler.timesteps

    # reset timesteps for ddim inference
    ddim_scheduler.set_timesteps(opt.steps)
    cond = get_text_embeds(tokenizer, text_encoder, opt.source_prompt, "", opt.device)[1].unsqueeze(0)
    latent_T = ddim_inversion(cond, latents, ddim_scheduler, unet, save_path, save_steps)
    print("DDIM inversion finished!")

    ######### Editing Guidance ###########################
    # save editing guidance mask
    set_editing_guidance(unet)
    register_target_block(unet, opt.target_block)
    register_target_layer_dim(unet, opt.target_layer_dim)
    register_target_prompt(unet, opt.target_prompt)
    register_target_words(unet, opt.target_words)
    register_edit_guidance_path(unet, save_edit_guidance_path)
    tgt_cond = get_text_embeds(tokenizer, text_encoder, opt.target_prompt, "", opt.device)[1].unsqueeze(0)

    latent_recon = ddim_sample_with_target(cond, latent_T, ddim_scheduler, unet, opt, tgt_cond)
    rgb_recon = decode_latents(vae, latent_recon)

    frames = (rgb_recon * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
    write_video(os.path.join(save_recon_video_path, f'recon_video.mp4'), frames, fps=fps)

if __name__ == "__main__":
    config_path = 'configs/config_sample2.yaml'
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    # from config
    parser.add_argument('--device', type=str, default=config['device']) 
    parser.add_argument('--input_video', type=str, default=config['input_video']) 
    parser.add_argument('--seed', type=str, default=config['seed']) 
    parser.add_argument('--source_prompt', type=str, default=config['source_prompt'])
    parser.add_argument('--target_prompt', type=str, default=config['target_prompt'])
    parser.add_argument('--target_block', type=str, default=config['target_block'])
    parser.add_argument('--target_layer_dim', type=str, default=config['target_layer_dim'])
    parser.add_argument('--target_step', type=str, default=config['target_step'])
    parser.add_argument('--target_words', type=list, default=config['target_words'])
    parser.add_argument('--out_dir', type=str, default=config['ddim_latents_path'])
    
    # static option
    parser.add_argument('--h', type=int, default=512)
    parser.add_argument('--w', type=int, default=512)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--save_steps', type=int, default=50)
    opt = parser.parse_args()
    print(f"source_prompt: {opt.source_prompt}")
    print(f"target_prompt: {opt.target_prompt}")
    opt.video_name = Path(opt.input_video).stem
    run(opt)
