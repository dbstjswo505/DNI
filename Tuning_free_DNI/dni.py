import glob
import os
import numpy as np
import cv2
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from PIL import Image
import yaml
from tqdm import tqdm
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionPipeline
import pdb

from dni_utils import *
from torchvision.io import read_video, write_video
import random

# suppress partial model loading warning
logging.set_verbosity_error()

VAE_BATCH_SIZE = 10

def seed_fix(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def save_video(raw_frames, save_path, fps=10):
    video_codec = "libx264"
    video_options = {"crf": "18", "preset": "slow",}
    frames = (raw_frames * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
    write_video(save_path, frames, fps=fps, video_codec=video_codec, options=video_options)


class DNI(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]

        ##################### Load stable diffusion ##############################
        sd_v = "stabilityai/stable-diffusion-2-1-base"
        self.pipe = StableDiffusionPipeline.from_pretrained(sd_v, torch_dtype=torch.float16).to("cuda")
        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(sd_v, subfolder="scheduler")
        self.scheduler.set_timesteps(config["n_timesteps"], device=self.device)

        ### Load video latent and total noise from (ddim latent - vae latent) ###
        self.ddim_latents_path = os.path.join(self.config["ddim_latents_path"], self.config["video_name"])
        video,_,meta = read_video(self.config["input_video"], output_format="TCHW")
        self.config['fps'] = meta['video_fps']
        frames = []
        for i in range(len(video)):
            image = T.ToPILImage()(video[i])
            image = image.resize((self.config["h"], self.config["w"]),  resample=Image.Resampling.LANCZOS)
            frame = image.convert('RGB')
            frames = frames + [frame]
        frames = torch.stack([T.ToTensor()(frame) for frame in frames]).to(torch.float16).to(self.device)
        self.config['vid_len'] = len(video)
        self.latents = self.encode_imgs(frames, deterministic=True).to(torch.float16).to(self.device)
        self.eps = self.get_ddim_eps(self.latents).to(torch.float16).to(self.device)

        ################### Load source prompt and target prompt  ###############
        self.target_embeds = self.get_text_embeds(config["target_prompt"], config["negative_prompt"])
        src_prompt_path = os.path.join(self.ddim_latents_path, 'source_prompt', 'source_prompt.txt')
        with open(src_prompt_path, 'r') as f:
            src_prompt = f.read()
        self.config["source_prompt"] = src_prompt
        self.source_embeds = self.get_text_embeds(src_prompt, src_prompt).chunk(2)[0]

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def encode_imgs(self, imgs, batch_size=VAE_BATCH_SIZE, deterministic=False):
        imgs = 2 * imgs - 1
        latents = []
        for i in range(0, len(imgs), batch_size):
            posterior = self.vae.encode(imgs[i:i + batch_size]).latent_dist
            latent = posterior.mean if deterministic else posterior.sample()
            latents.append(latent * 0.18215)
        latents = torch.cat(latents)
        return latents

    @torch.no_grad()
    def decode_latents(self, latents, batch_size=VAE_BATCH_SIZE):
        latents = 1 / 0.18215 * latents
        imgs = []
        for i in range(0, len(latents), batch_size):
            imgs.append(self.vae.decode(latents[i:i + batch_size]).sample)
        imgs = torch.cat(imgs)
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def get_ddim_eps(self, latent):
        noisest = max([int(x.split('_')[-1].split('.')[0]) for x in glob.glob(os.path.join(self.ddim_latents_path, 'latents', f'ddim_latents_*.pt'))])
        latents_path = os.path.join(self.ddim_latents_path, 'latents', f'ddim_latents_{noisest}.pt')
        noisy_latent = torch.load(latents_path).to(self.device)
        alpha_prod_T = self.scheduler.alphas_cumprod[noisest]
        mu_T, sigma_T = alpha_prod_T ** 0.5, (1 - alpha_prod_T) ** 0.5
        eps = (noisy_latent - mu_T * latent) / sigma_T
        return eps

    @torch.no_grad()
    def denoise_step(self, x, t, indices):
        # register the time step and features in pnp injection modules
        latents_t_path = os.path.join(self.ddim_latents_path, 'latents', f'ddim_latents_{t}.pt')
        source_latents_ = torch.load(latents_t_path)[indices]
       
        # exp 1. Early DNI
        #mask = self.load_mask(source_latents_.dtype)[indices]
        #asf = ASF(self.latents)
        #asf = torch.transpose(asf, 0, 1)
        #asf = asf[indices]
        #source_latents = Dilution(source_latents_, asf, mask)
        source_latents = source_latents_
        init_latents = self.latents[indices]
        
        latent_model_input = torch.cat([init_latents] + [source_latents] + ([x] * 2))

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.source_embeds.repeat(len(indices), 1, 1),
                                      self.source_embeds.repeat(len(indices), 1, 1),
                                      torch.repeat_interleave(self.target_embeds, len(indices), dim=0)])
        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        _,_, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(4)
        noise_pred = noise_pred_uncond + self.config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    @torch.autocast(dtype=torch.float16, device_type='cuda')
    def batched_denoise_step(self, x, t, indices):
        batch_size = self.config["batch_size"]
        denoised_latents = []

        if self.config["module"] == 'propagation':
            key_idx = torch.randint(batch_size, (len(x)//batch_size,)) + torch.arange(0,len(x),batch_size)
            register_propagation(self, True, key_idx)
            self.denoise_step(x[key_idx], t, indices[key_idx])
            register_propagation(self, False)
        
        for i, b in enumerate(range(0, len(x), batch_size)):
            register_batch_idx(self, i)
            register_propagation(self, False, indices[b:b + batch_size])
            denoised_latents.append(self.denoise_step(x[b:b + batch_size], t, indices[b:b + batch_size]))

        denoised_latents = torch.cat(denoised_latents)
        return denoised_latents

    def register_dni(self, apply_timesteps):
        register_attn_dni(self, apply_timesteps)
        register_conv_dni(self, apply_timesteps)

    def load_mask(self, f_dtype):
        mask_rgd = []
        mask_non_rgd = []
        for i in range(len(self.config['target_words'])):
            word = self.config['target_words'][i]
            mask = np.load(os.path.join(config['edit_guidance_path'],f'{word}.npy'))
            if word in config['rgd']:
                mask_rgd = mask_rgd + [mask]
            else:
                mask_non_rgd = mask_non_rgd + [mask]
        if len(mask_rgd) != 0:
            mask_rgd = np.stack(mask_rgd, axis=-1)
            mask_rgd = torch.from_numpy(mask_rgd)
            mask_rgd = mask_rgd.to(f_dtype)
            mask_rgd = mask_rgd.mean(dim=-1)
            mask_rgd = mask_rgd/255
        
        if len(mask_non_rgd) != 0:
            mask_non_rgd = np.stack(mask_non_rgd, axis=-1)
            mask_non_rgd = torch.from_numpy(mask_non_rgd)
            mask_non_rgd = mask_non_rgd.to(f_dtype)
            mask_non_rgd = mask_non_rgd.mean(dim=-1)
            mask_non_rgd = mask_non_rgd/255
            
        if len(mask_non_rgd) == 0:
            out_mask = mask_rgd*self.config['alpha']
        elif len(mask_rgd) == 0:
            out_mask = mask_non_rgd*self.config['beta']
        else:
            out_mask = mask_rgd*self.config['alpha'] + mask_non_rgd*self.config['beta']
        return out_mask

    def edit_video(self):
        apply_t = int(self.config["n_timesteps"] * self.config["dni_apply_step"])
        apply_timesteps = self.scheduler.timesteps[:apply_t]
        self.register_dni(apply_timesteps)

        register_temporal_module(self.unet, self.config["module"])

        asf_filter = ASF(self.latents)
        eg_mask = self.load_mask(self.latents.dtype)
        register_edit_guidance_mask(self, eg_mask, self.config["gamma"])

        noisy_latents = self.scheduler.add_noise(self.latents, self.eps, self.scheduler.timesteps[0])
        noisy_latents = Dilution(noisy_latents, asf_filter, eg_mask, self.config["gamma"])

        edited_frames = self.sample_loop(noisy_latents, torch.arange(self.config["n_frames"]))
        save_video(edited_frames, f'{self.config["output_path"]}/{self.config["video_name"]}.mp4', fps=self.config["fps"])
        print('Done!')

    def sample_loop(self, x, indices):
        os.makedirs(f'{self.config["output_path"]}', exist_ok=True)
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
            x = self.batched_denoise_step(x, t, indices)
        
        decoded_latents = self.decode_latents(x)
        return decoded_latents

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_sample2.yaml')
    opt = parser.parse_args()
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
    video_name = Path(config["input_video"]).stem
    config["output_path"] = os.path.join(config["output_path"], video_name, config["target_prompt"])
    config["video_name"] = video_name
    config["edit_guidance_path"] = os.path.join(config["ddim_latents_path"], video_name, 'edit_guidance')
    os.makedirs(config["output_path"], exist_ok=True)

    target_prompt = config["target_prompt"]
    print(f"Target prompt: {target_prompt}")
    
    seed_fix(config["seed"])
    editor = DNI(config)
    editor.edit_video()

