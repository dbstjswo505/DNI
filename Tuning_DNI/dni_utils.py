import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import DDIMScheduler
import cv2
import pdb
import os
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
from einops import rearrange
from torch.optim.adam import Adam
import torch.fft as fft
import torch.nn.functional as nnf

NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def isinstance_str(x: object, cls_name: str):
    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    return False

def load_512_seq(image_path, left=0, right=0, top=0, bottom=0, n_sample_frame=10, sampling_rate=1):
    images = []
    for file in sorted(os.listdir(image_path)):
        images.append(file)
    n_images = len(images)
    sequence_length = (n_sample_frame - 1) * sampling_rate + 1
    if n_images < sequence_length:
        raise ValueError
    frames = []
    for index in range(n_sample_frame):
        p = os.path.join(image_path, images[index*sampling_rate])
        image = np.array(Image.open(p).convert("RGB"))
        h, w, c = image.shape
        left = min(left, w-1)
        right = min(right, w - left - 1)
        top = min(top, h - left - 1)
        bottom = min(bottom, h - top - 1)
        image = image[top:h-bottom, left:w-right]
        h, w, c = image.shape
        if h < w:
            offset = (w - h) // 2
            image = image[:, offset:offset + h]
        elif w < h:
            offset = (h - w) // 2
            image = image[offset:offset + w]
        image = np.array(Image.fromarray(image).resize((512, 512)))
        frames.append(image)
    return np.stack(frames)

def ASF(latent_0):
    # input: latent_0
    # output: adaptive spectral filter

    L,C,W,H = latent_0.shape
    latent_0 = latent_0.reshape(-1, L, W, H)

    # 1. 3D-Discrete Fourier Transform
    latent_0_freq = fft.fftn(latent_0.to(torch.float64), dim=(-3, -2, -1))
    latent_0_freq = fft.fftshift(latent_0_freq, dim=(-3, -2, -1))
    latent_0_freq_mag = torch.log(torch.abs(latent_0_freq)+1)

    # 2. min-max normalization
    latent_0_freq_max, _ = torch.max(latent_0_freq_mag.view(C,-1), dim=-1, keepdim=True)
    latent_0_freq_min, _ = torch.min(latent_0_freq_mag.view(C,-1), dim=-1, keepdim=True)
    latent_0_freq_max = latent_0_freq_max.view(C,1,1,1)
    latent_0_freq_min = latent_0_freq_min.view(C,1,1,1)
    freq_filter = (latent_0_freq_mag - latent_0_freq_min)/(latent_0_freq_max - latent_0_freq_min)
    return freq_filter

def Dilution(latent_0, freq_filter, guidance_mask, gamma=1.0):
    # input: latent, latent_0, guidance_mask
    # output: dilutional_latent
    latent = latent_0.squeeze()
    C,L,W,H = latent.shape
    # Noise Disentanglement
    # 1. Build adaptive spectral filter (f)
    f = freq_filter

    # 2. Disentangling visual noise and Gaussian noise
    f_dtype = latent.dtype
    #latent_freq = fft.fftn(latent.to(torch.complex128), dim=(-3, -2, -1))
    latent_freq = fft.fftn(latent.to(torch.float64), dim=(-3, -2, -1))
    latent_freq = fft.fftshift(latent_freq, dim=(-3, -2, -1))

    visual_freq = f * latent_freq
    noise_freq = (1 - f) * latent_freq

    visual_freq = fft.ifftshift(visual_freq, dim=(-3, -2, -1))
    visual_noise = fft.ifftn(visual_freq, dim=(-3, -2, -1)).real
    visual_branch = visual_noise.to(f_dtype)

    noise_freq = fft.ifftshift(noise_freq, dim=(-3, -2, -1))
    Gaussian_noise = fft.ifftn(noise_freq, dim=(-3, -2, -1)).real
    noise_branch = Gaussian_noise.to(f_dtype)

    # Noise Dilution
    # 1. min-max normalization: mask = min_max_norm(guidance_mask)
    m_min, _ = torch.min(guidance_mask.view(-1,L), dim=0)
    m_max, _ = torch.max(guidance_mask.view(-1,L), dim=0)
    m_min = m_min.view(-1,1,1)
    m_max = m_max.view(-1,1,1)
    mask = (guidance_mask - m_min)/(m_max - m_min + 0.0000001)
    mask = mask.view(1,-1,W,H).repeat(C,1,1,1)
    mask = mask.to(latent.device)

    # 2. Gaussian random noise
    white_gaussian = torch.randn(visual_branch.shape, device=latent.device, dtype=latent.dtype)
    # 3. Blending
    dilutional_latent = noise_branch * gamma + (white_gaussian * mask + visual_branch * (1-mask)) * (2-gamma)

    return dilutional_latent.unsqueeze(0)


class NullInversion:

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def latent2image_video(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        latents = latents[0].permute(1, 0, 2, 3)
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device, dtype=self.model.unet.dtype)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def image2latent_video(self, image):
        with torch.no_grad():
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(0, 3, 1, 2).to(device).to(device, dtype=self.model.unet.dtype)
            latents = self.model.vae.encode(image)['latent_dist'].mean
            latents = rearrange(latents, "(b f) c h w -> b c f h w", b=1)
            latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        register_target_save(self.model, False)
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            if t == self.model.unet.target_step:
                register_target_save(self.model, True)
                noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
                register_target_save(self.model, False)
            else:
                noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent_video(image)
        image_rec = self.latent2image_video(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents
    
    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings) # predict additive noise from current latent
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                # bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break

            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        # bar.close()
        return uncond_embeddings_list

    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), video_len=24, num_inner_steps=10, early_stop_epsilon=1e-5, sampling_rate=1):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        image_gt = load_512_seq(image_path, *offsets, video_len, sampling_rate)
        print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], ddim_latents[0], uncond_embeddings

    def invert_fast(self, image_path: str, prompt: str, offsets=(0,0,0,0), video_len=24, num_inner_steps=10, early_stop_epsilon=1e-5, sampling_rate=1):
        self.init_prompt(prompt)
        register_attention_control(self.model, None) # Nothing to work
        image_gt = load_512_seq(image_path, *offsets, video_len, sampling_rate)
        print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        return (image_gt, image_rec), ddim_latents[-1], ddim_latents[0], None

    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)


def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False, simple=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    if simple:
        noise_pred[0] = noise_prediction_text[0]
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


@torch.no_grad()
def latent2image_video(vae, latents):
    latents = 1 / 0.18215 * latents
    latents = latents[0].permute(1, 0, 2, 3)
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)
    
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]
    
    text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])
    
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)
    
    image = latent2image(model.vqvae, latents)
   
    return image, latent


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    # set timesteps
    extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
    
    image = latent2image(model.vae, latents)
  
    return image, latent


def register_target_step(model, target_step):
    module = model.unet
    setattr(module, 'target_step', target_step)

def register_target_block(model, target_block):
    for _, module in model.unet.named_modules():
        if isinstance_str(module, "CrossAttention"):
            setattr(module, "target_block", False)
    up_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    down_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    A = False
    B = False
    C = False
    if target_block == 'up':
        A = True
    elif target_block == 'down':
        B = True
    elif target_block == 'mid':
        C = True
    for res in up_dict:
        for block in up_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'target_block', A)
    for res in down_dict:
        for block in down_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'target_block', B)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 'target_block', C)

def register_target_words(model, target_words):
    for _, module in model.unet.named_modules():
        if isinstance_str(module, "CrossAttention"):
            setattr(module, "target_words", target_words)

def register_target_prompt(model, target_prompt):
    for _, module in model.unet.named_modules():
        if isinstance_str(module, "CrossAttention"):
            setattr(module, "target_prompt", target_prompt)

def register_edit_guidance_path(model, edit_guidance_path):
    for _, module in model.unet.named_modules():
        if isinstance_str(module, "CrossAttention"):
            setattr(module, "edit_guidance_path", edit_guidance_path)

def register_target_layer_dim(model, target_layer_dim):
    for _, module in model.unet.named_modules():
        if isinstance_str(module, "CrossAttention"):
            setattr(module, "target_layer_dim", target_layer_dim)

def register_target_save(model, target_save):
    for _, module in model.unet.named_modules():
        if isinstance_str(module, "CrossAttention"):
            setattr(module, "target_save", target_save)

def register_edit_guidance_mask(model, eg_mask):
    module = model.unet
    setattr(module, 'eg_mask', eg_mask)

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            context = encoder_hidden_states
            mask = attention_mask
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = torch.exp(sim-torch.max(sim)) / torch.sum(torch.exp(sim-torch.max(sim)), axis=-1).unsqueeze(-1)
            #: Use Attention Blend or not
            if controller.__class__.__name__ == 'DummyController':
                if self.target_block and x.shape[-1] == self.target_layer_dim and self.target_save:
                    _,seqlen,txtlen = attn.shape
                    attn_out = attn.view(-1,h,seqlen,txtlen)
                    attn_out = attn_out.mean(dim=1)
                    attmap = torch.transpose(attn_out, 1, 2)
                    tgt_list = self.target_prompt.split(' ')
                    for i in range(len(self.target_words)):
                        word = self.target_words[i]
                        tgt_idx = tgt_list.index(word) + 1 # +1 is <sos>
                        tgt_attmap = attmap[:,tgt_idx,:]
                        w = int((tgt_attmap.shape[-1])**0.5)
                        tgt_attmap = tgt_attmap.reshape(-1,w,w).cpu().numpy()

                        # visual version (using raw data can be more precise)
                        tgt_attmap_vis = (tgt_attmap)*255
                        tgt_attmap_vis = tgt_attmap_vis.astype(np.uint8)
                        data = []
                        for j in range(tgt_attmap.shape[0]):
                            img = Image.fromarray(tgt_attmap_vis[j])
                            img = img.resize((64,64), Image.BICUBIC)
                            img.save(self.edit_guidance_path+f'/{word}_{j}.jpg')
                            data = data + [np.array(img)]
                        data = np.stack(data, axis=0)
                        np.save(self.edit_guidance_path+f'/{word}', data)

                    print("editing guidance mask is saved")
            else:
                #pdb.set_trace()
                z=2
            attn = controller(attn, is_cross, place_in_unet) # attn: 64, L_p, L_t
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count

    
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1): # 2
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"], # {'default_': 0.8}
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words
