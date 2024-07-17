# Adapted from https://github.com/google/prompt-to-prompt/blob/main/null_text_w_ptp.ipynb

import os
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import AutoencoderKL, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import dni_utils
from dni_utils import NullInversion
import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer
from einops import rearrange

from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline

import cv2
import argparse
from omegaconf import OmegaConf
import pdb

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
LOW_RESOURCE = False
NUM_DDIM_STEPS = 50
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# need to adjust sometimes
mask_th = (.3, .3)

def main(
    pretrained_model_path: str,
    image_path: str,
    prompt: str,
    prompts: Tuple[str],
    eq_params: Dict,
    save_name: str,
    is_word_swap: bool = True,
    target_step: int = None,
    target_block: str = None,
    target_layer_dim: int = None,
    target_words: List[str] = None,
    alpha: float = None,
    beta: float = None,
    gamma: float = None,
    rgd: List[str] = None,
    blend_word: Tuple[str] = None,
    cross_replace_steps: float = 0.2,
    self_replace_steps: float = 0.5,
    video_len: int = 24,
    fast: bool = False,
    mixed_precision: Optional[str] = 'fp32',
    sample_frame_rate: int = 1,
):
    output_folder = os.path.join(pretrained_model_path, 'results')
    output_edit_guidance_folder = os.path.join(pretrained_model_path, 'edit_guidance')
    if fast:
        save_name_1 = os.path.join(output_folder, 'inversion_fast.gif')
        save_name_2 = os.path.join(output_folder, '{}_fast.gif'.format(save_name))
    else:
        save_name_1 = os.path.join(output_folder, 'inversion.gif')
        save_name_2 = os.path.join(output_folder, '{}.gif'.format(save_name))
    if blend_word:
        blend_word = (((blend_word[0],), (blend_word[1],)))
    eq_params = dict(eq_params)
    prompts = list(prompts)
    cross_replace_steps = {'default_': cross_replace_steps,}
    
    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_edit_guidance_folder):
        os.makedirs(output_edit_guidance_folder)

    # Load the tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path,
        subfolder="text_encoder",
    ).to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_path,
        subfolder="vae",
    ).to(device, dtype=weight_dtype)
    unet = UNet3DConditionModel.from_pretrained(
        pretrained_model_path, subfolder="unet"
    ).to(device, dtype=weight_dtype)
    ldm_stable = TuneAVideoPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    ).to(device)

    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    tokenizer = ldm_stable.tokenizer # Tokenizer of class: [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer)
    # A tokenizer breaks a stream of text into tokens, usually by looking for whitespace (tabs, spaces, new lines).

    class LocalBlend:
        
        def get_mask(self, maps, alpha, use_pool):
            k = 1
            maps = (maps * alpha).sum(-1).mean(2)
            if use_pool:
                maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
            mask = nnf.interpolate(maps, size=(x_t.shape[3:]))
            mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
            mask = mask.gt(self.th[1-int(use_pool)])
            mask = mask[:1] + mask
            return mask
        
        def __call__(self, x_t, attention_store, step):
            self.counter += 1
            if self.counter > self.start_blend:
                maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
                maps = [item.reshape(self.alpha_layers.shape[0], -1, 8, 16, 16, MAX_NUM_WORDS) for item in maps]
                maps = torch.cat(maps, dim=2)
                mask = self.get_mask(maps, self.alpha_layers, True)
                if self.substruct_layers is not None:
                    maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                    mask = mask * maps_sub
                mask = mask.float()
                mask = mask.reshape(-1, 1, mask.shape[-3], mask.shape[-2], mask.shape[-1])
                x_t = x_t[:1] + mask * (x_t - x_t[:1])
            return x_t
        
            # attention_store.keys() = dict_keys(['down_cross', 'mid_cross', 'up_cross', 'down_self', 'mid_self', 'up_self'])
            # attention_store['up_cross'] => torch.Size([128, 1024, 77]) 4개 담겨있음
            # attention_store['down_cross'] => torch.Size([128, 1024, 77]) 4개 담겨있음
            
        
        def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2, th=(.3, .3)):
            alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = dni_utils.get_word_inds(prompt, word, tokenizer)
                    alpha_layers[i, :, :, :, :, ind] = 1
            
            if substruct_words is not None:
                substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
                for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                    if type(words_) is str:
                        words_ = [words_]
                    for word in words_:
                        ind = dni_utils.get_word_inds(prompt, word, tokenizer)
                        substruct_layers[i, :, :, :, :, ind] = 1
                self.substruct_layers = substruct_layers.to(device)
            else:
                self.substruct_layers = None
            self.alpha_layers = alpha_layers.to(device)
            self.start_blend = int(start_blend * NUM_DDIM_STEPS)
            self.counter = 0 
            self.th=th
            
            
    class EmptyControl:
        
        
        def step_callback(self, x_t):
            return x_t
        
        def between_steps(self):
            return
        
        def __call__(self, attn, is_cross: bool, place_in_unet: str):
            return attn

        
    class AttentionControl(abc.ABC):
        
        def step_callback(self, x_t):
            return x_t
        
        def between_steps(self):
            return
        
        @property
        def num_uncond_att_layers(self):
            return self.num_att_layers if LOW_RESOURCE else 0
        
        @abc.abstractmethod
        def forward (self, attn, is_cross: bool, place_in_unet: str): # 이걸 상속하려면 반드시 이 함수를 상속받는 클래스에서도 선언해주어야 함!
            raise NotImplementedError

        def __call__(self, attn, is_cross: bool, place_in_unet: str):
            if self.cur_att_layer >= self.num_uncond_att_layers:
                if LOW_RESOURCE:
                    attn = self.forward(attn, is_cross, place_in_unet)
                else:
                    h = attn.shape[0]
                    attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
            self.cur_att_layer += 1
            if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
                self.cur_att_layer = 0
                self.cur_step += 1
                self.between_steps()
            return attn
        
        def reset(self):
            self.cur_step = 0
            self.cur_att_layer = 0

        def __init__(self):
            self.cur_step = 0
            self.num_att_layers = -1
            self.cur_att_layer = 0

    class SpatialReplace(EmptyControl):
        
        def step_callback(self, x_t):
            if self.cur_step < self.stop_inject:
                b = x_t.shape[0]
                x_t = x_t[:1].expand(b, *x_t.shape[1:])
            return x_t

        def __init__(self, stop_inject: float):
            super(SpatialReplace, self).__init__()
            self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)
            

    class AttentionStore(AttentionControl):

        @staticmethod
        def get_empty_store():
            return {"down_cross": [], "mid_cross": [], "up_cross": [],
                    "down_self": [],  "mid_self": [],  "up_self": []}

        def forward(self, attn, is_cross: bool, place_in_unet: str): # forward(attn, is_cross, place_in_unet)
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            if attn.shape[1] <= 32 ** 2:
                self.step_store[key].append(attn)
            return attn     # attn.shape = torch.Size([16384, 8, 8])

        def between_steps(self):
            if len(self.attention_store) == 0:
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] += self.step_store[key][i]
            self.step_store = self.get_empty_store()

        def get_average_attention(self):
            average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
            return average_attention


        def reset(self):
            super(AttentionStore, self).reset()
            self.step_store = self.get_empty_store()
            self.attention_store = {}

        def __init__(self):
            super(AttentionStore, self).__init__()
            self.step_store = self.get_empty_store()
            self.attention_store = {}

            
    class AttentionControlEdit(AttentionStore, abc.ABC):
        
        # NOTE: 여기다!! pieline_tuneavideo.py에서 latents = controller.step_callback(latents).to(device, dtype=weight_type) 로 들어옴
        def step_callback(self, x_t):
            if self.local_blend is not None:
                x_t = self.local_blend(x_t, self.attention_store, self.cur_step)
            return x_t
            
        def replace_self_attention(self, attn_base, att_replace, place_in_unet):
            if att_replace.shape[2] <= 32 ** 2:
                attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
                return attn_base
            else:
                return att_replace
        
        @abc.abstractmethod
        def replace_cross_attention(self, attn_base, att_replace):
            raise NotImplementedError
        
        # NOTE: 여기다!! 여기에서 attention_store의 forward() 가 실행됨.
        def get_attn(self, attn, place_in_unet, step):
            b, Lp, Lt = attn.shape
            name ='step_'+ str(step) + '_' + place_in_unet + '_' + str(Lp)
            out_pth = './attn_out/'
            import copy
            tmp = copy.deepcopy(attn)
            np.save(out_pth+name, tmp.cpu().numpy()) 

        def forward(self, attn, is_cross: bool, place_in_unet: str):
            super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
            if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
                h = attn.shape[0] // (self.batch_size)
                attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
                attn_base, attn_replace = attn[0], attn[1:]
                if is_cross:
                    # attention save 
                    #self.get_attn(attn[1], place_in_unet, self.cur_step)
                    # blending
                    alpha_words = self.cross_replace_alpha[self.cur_step]
                    #attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                    attn_replace_new = self.replace_cross_attention(attn_base, attn_replace)
                    attn[1:] = attn_replace_new
                    z=1
                else:
                    # blending
                    z=1
                    #attn[1:] = self.replace_self_attention(attn_base, attn_replace, place_in_unet)
                attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
            return attn
        
        def __init__(self, prompts, num_steps: int,
                    cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                    self_replace_steps: Union[float, Tuple[float, float]],
                    local_blend: Optional[LocalBlend]):
            super(AttentionControlEdit, self).__init__()
            self.batch_size = len(prompts)
            self.cross_replace_alpha = dni_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
            if type(self_replace_steps) is float:
                self_replace_steps = 0, self_replace_steps
            self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
            self.local_blend = local_blend

    class AttentionReplace(AttentionControlEdit):

        def replace_cross_attention(self, attn_base, att_replace):
            return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
        
        def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                    local_blend: Optional[LocalBlend] = None):
            super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
            #self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
            

    class AttentionRefine(AttentionControlEdit):

        def replace_cross_attention(self, attn_base, att_replace):
            attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
            attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
            return attn_replace

        def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                    local_blend: Optional[LocalBlend] = None):
            super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
            #self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
            #self.mapper, alphas = self.mapper.to(device), alphas.to(device)
            #self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


    class AttentionReweight(AttentionControlEdit):

        def replace_cross_attention(self, attn_base, attn_replace):
            #if self.prev_controller is not None:
                 #This is attention controll about replace or refine
                 #attn_base = self.prev_controller.replace_cross_attention(attn_base, attn_replace)
            #attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
            attn_replace = attn_replace[None, :, :, :] * self.equalizer[:, None, None, :]
            return attn_replace

        def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                    local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
            super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
            self.equalizer = equalizer.to(device)
            self.prev_controller = controller


    def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                    Tuple[float, ...]]):
        if type(word_select) is int or type(word_select) is str:
            word_select = (word_select,)
        equalizer = torch.ones(1, 77)
        for word, val in zip(word_select, values):
            inds = dni_utils.get_word_inds(text, word, tokenizer)
            equalizer[:, inds] = val
        return equalizer

    def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
        out = []
        attention_maps = attention_store.get_average_attention()
        num_pixels = res ** 2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(8, 8, res, res, item.shape[-1])
                    out.append(cross_maps)
        out = torch.cat(out, dim=1)
        out = out.sum(1) / out.shape[1]
        return out.cpu()


    def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, blend_words=None, equilizer_params=None, mask_th=(.3,.3)) -> AttentionControlEdit:
        if blend_words is None:
            lb = None
        else:
            lb = LocalBlend(prompts, blend_word, th=mask_th)
        if is_replace_controller:
            controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
        else:
            controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
        if equilizer_params is not None:
            eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"])
            controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller)
        return controller

    def load_mask(f_dtype, target_words, edit_guidance_path, rgd, alpha, beta):
        mask_rgd = []
        mask_non_rgd = []
        for i in range(len(target_words)):
            word = target_words[i]
            mask = np.load(os.path.join(edit_guidance_path,f'{word}.npy'))
            if word in rgd:
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
            out_mask = mask_rgd*alpha
        elif len(mask_rgd) == 0:
            out_mask = mask_non_rgd*beta
        else:
            out_mask = mask_rgd*alpha + mask_non_rgd*beta
        return out_mask

    null_inversion = NullInversion(ldm_stable)
    dni_utils.register_target_layer_dim(ldm_stable, target_layer_dim)
    dni_utils.register_target_block(ldm_stable, target_block)
    dni_utils.register_target_step(ldm_stable, target_step)
    dni_utils.register_target_prompt(ldm_stable, prompt)
    dni_utils.register_target_words(ldm_stable, target_words)
    dni_utils.register_edit_guidance_path(ldm_stable, output_edit_guidance_folder)

    ###############
    # Custom APIs:

    ldm_stable.enable_xformers_memory_efficient_attention()
    if fast:
        (image_gt, image_enc), x_t, x_0, uncond_embeddings = null_inversion.invert_fast(image_path, prompt, offsets=(0,0,0,0), video_len=video_len, sampling_rate=sample_frame_rate)
    else:
        (image_gt, image_enc), x_t, x_0, uncond_embeddings = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), video_len=video_len, sampling_rate=sample_frame_rate)
    
    eg_mask = load_mask(x_t.dtype, target_words, output_edit_guidance_folder, rgd, alpha, beta)

    x_0 = x_0.squeeze().transpose(0,1).contiguous()
    asf_filter = dni_utils.ASF(x_0)
    x_t = dni_utils.Dilution(x_t, asf_filter, eg_mask, gamma)

    print("Start Video-P2P!")
    
    #NOTE: make_controller
    controller = make_controller(prompts, is_word_swap, cross_replace_steps, self_replace_steps, blend_word, eq_params, mask_th=mask_th)
    dni_utils.register_attention_control(ldm_stable, controller)
    generator = torch.Generator(device=device)
    with torch.no_grad():
        sequence = ldm_stable(
            prompts,
            generator=generator,
            latents=x_t,
            uncond_embeddings_pre=uncond_embeddings,
            controller = controller,
            video_length=video_len,
            fast=fast,
        ).videos
    
    seq_len = len(sequence)
    svpth = save_name_1.split('/')
    fix_pth = '.'
    for arg in svpth[1:-1]:
        fix_pth = os.path.join(fix_pth, arg)
    for i in range(seq_len):
        sequence1 = rearrange(sequence[i], "c t h w -> t h w c")
        inversion = []
        #videop2p = []
        
        for j in range(sequence1.shape[0]):
            inversion.append(Image.fromarray((sequence1[j] * 255).numpy().astype(np.uint8)) )
            #videop2p.append(Image.fromarray((sequence2[i] * 255).numpy().astype(np.uint8)) )

        #duration = 1000/fps
        out_pth = fix_pth + '/' + prompts[i] +'.gif'
        inversion[0].save(out_pth, save_all=True, append_images=inversion[1:], optimize=False, loop=0, duration=125)
        #videop2p[0].save(save_name_2, save_all=True, append_images=videop2p[1:], optimize=False, loop=0, duration=125)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default = "./configs/snowboard/snowboard-p2p.yaml") #default="./configs/videop2p.yaml")
    parser.add_argument("--fast", action='store_true')
    args = parser.parse_args()

    main(**OmegaConf.load(args.config), fast=args.fast)
