from typing import Type
import torch
import os
import pdb
import numpy as np
import copy
from PIL import Image
import torch.fft as fft
import math
import pdb

def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.

    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True

    return False


def batch_cosine_sim(x, y):
    if type(x) is list:
        x = torch.cat(x, dim=0)
    if type(y) is list:
        y = torch.cat(y, dim=0)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    similarity = x @ y.T
    return similarity

def register_propagation(model, is_propagation, key_idx=None):
    for _, module in model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "propagation_pass", is_propagation)
            setattr(module, "key_idx", key_idx)
    
    up_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    down_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}

    for res in up_dict:
        for block in up_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'propagation_pass', is_propagation)
            setattr(module, "key_idx", key_idx)
    for res in down_dict:
        for block in down_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'propagation_pass', is_propagation)
            setattr(module, "key_idx", key_idx)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 'propagation_pass', is_propagation)
    setattr(module, "key_idx", key_idx)
    
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 'propagation_pass', is_propagation)
    setattr(conv_module, "key_idx", key_idx)
            
def register_batch_idx(diffusion_model, batch_idx):
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "batch_idx", batch_idx)

def register_target_block(model, target_block):
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
            module = model.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'target_block', A)
    for res in down_dict:
        for block in down_dict[res]:
            module = model.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'target_block', B)
    module = model.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 'target_block', C)
        
def register_target_layer_dim(model, target_layer_dim):
    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "target_layer_dim", target_layer_dim)

def register_target_save(model, target_save):
    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "target_save", target_save)

def register_target_prompt(model, target_prompt):
    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "target_prompt", target_prompt)

def register_target_words(model, target_words):
    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "target_words", target_words)

def register_edit_guidance_path(model, edit_guidance_path):
    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "edit_guidance_path", edit_guidance_path)

def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 't', t)

def register_filter(model, asf_filter):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 'asf_filter', asf_filter)

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "asf_filter", asf_filter)

def register_edit_guidance_mask(model, eg_mask, gamma=1.0):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 'eg_mask', eg_mask)
    setattr(conv_module, 'gamma', gamma)

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "eg_mask", eg_mask)
            setattr(module, 'gamma', gamma)
    
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'eg_mask', eg_mask)
            setattr(module, 'gamma', gamma)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'eg_mask', eg_mask)
            setattr(module, 'gamma', gamma)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'eg_mask', eg_mask)
            setattr(module, 'gamma', gamma)
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'eg_mask', eg_mask)
            setattr(module, 'gamma', gamma)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 'eg_mask', eg_mask)
    setattr(module, 'gamma', gamma)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 'eg_mask', eg_mask)
    setattr(module, 'gamma', gamma)

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

def Dilution(latent, freq_filter, guidance_mask, gamma=1.0):
    # input: latent, latent_0, guidance_mask
    # output: dilutional_latent
    L,C,W,H = latent.shape
    #latent = latent.reshape(-1,L,W,H)
    latent = latent.transpose(0,1).contiguous()
    # Noise Disentanglement
    # 1. Build adaptive spectral filter (f)
    f = freq_filter

    # 2. Disentangling visual noise and Gaussian noise
    f_dtype = latent.dtype
    latent_freq = fft.fftn(latent.to(torch.float64), dim=(-3, -2, -1))
    latent_freq = fft.fftshift(latent_freq, dim=(-3, -2, -1))
    visual_freq = f * latent_freq
    noise_freq = (1 - f) * latent_freq

    visual_freq = fft.ifftshift(visual_freq, dim=(-3, -2, -1))
    visual_noise = fft.ifftn(visual_freq, dim=(-3, -2, -1)).real
    visual_branch = visual_noise.to(f_dtype)
    visual_branch = visual_branch.transpose(0,1).contiguous()
    #visual_branch = visual_branch.view(L,C,W,H)

    noise_freq = fft.ifftshift(noise_freq, dim=(-3, -2, -1))
    Gaussian_noise = fft.ifftn(noise_freq, dim=(-3, -2, -1)).real
    noise_branch = Gaussian_noise.to(f_dtype)
    noise_branch = noise_branch.transpose(0,1).contiguous()
    #noise_branch = noise_branch.view(L,C,W,H)
    
    # Noise Dilution
    # 1. min-max normalization: mask = min_max_norm(guidance_mask)
    m_min, _ = torch.min(guidance_mask.view(-1,L), dim=0)
    m_max, _ = torch.max(guidance_mask.view(-1,L), dim=0)
    m_min = m_min.view(-1,1,1)
    m_max = m_max.view(-1,1,1)
    mask = (guidance_mask - m_min)/(m_max - m_min + 0.000001)
    mask = mask.view(-1,1,W,H).repeat(1,C,1,1)
    mask = mask.to(latent.device)

    # 2. Gaussian random noise
    white_gaussian = torch.randn(visual_branch.shape, device=latent.device, dtype=latent.dtype)
    # 3. Blending
    dilutional_latent = noise_branch * gamma + (white_gaussian * mask + visual_branch * (1-mask)) * (2-gamma)
    #dilutional_latent = visual_branch

    return dilutional_latent


def register_conv_dni(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 4)
                init_hidden_states = hidden_states[:source_batch_size]
                asf = ASF(init_hidden_states)
                
                eg_mask = self.eg_mask[self.key_idx]
                eg_mask = eg_mask.unsqueeze(1)
                L,_,W,H = eg_mask.shape
                _,_,w,h = init_hidden_states.shape
                eg_mask = torch.nn.functional.avg_pool2d(eg_mask.to(torch.float64), int(W/w), int(W/w))
                eg_mask = eg_mask.to(init_hidden_states.dtype)
                eg_mask = eg_mask.squeeze(1)

                dilute_hidden_states = Dilution(hidden_states[source_batch_size:2 * source_batch_size], asf, eg_mask, self.gamma)

                # dilutional inject unconditional
                hidden_states[2 * source_batch_size:3 * source_batch_size] = dilute_hidden_states

                # dilutional inject conditional
                hidden_states[3 * source_batch_size:] = dilute_hidden_states


            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)

def register_attn_dni(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            n_frames = batch_size // 4
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)

            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                _,sql,cn = q.shape
                q_hidden_states = torch.transpose(q, 1, 2).reshape(-1, cn, int(sql**0.5), int(sql**0.5))
                k_hidden_states = torch.transpose(k, 1, 2).reshape(-1, cn, int(sql**0.5), int(sql**0.5))
                q_init_hidden_states = q_hidden_states[:n_frames]
                k_init_hidden_states = k_hidden_states[:n_frames]

                qasf = ASF(q_init_hidden_states)
                
                kasf = ASF(k_init_hidden_states)

                eg_mask = self.eg_mask[self.key_idx]
                eg_mask = eg_mask.unsqueeze(1)
                L,_,sqW,sqH = eg_mask.shape
                _,_,sqw,sqh = q_init_hidden_states.shape
                eg_mask = torch.nn.functional.avg_pool2d(eg_mask.to(torch.float64), int(sqW/sqw), int(sqW/sqw))
                eg_mask = eg_mask.to(q_init_hidden_states.dtype)
                eg_mask = eg_mask.squeeze(1)

                q_dilute_hidden_states = Dilution(q_hidden_states[n_frames:2 * n_frames], qasf, eg_mask, self.gamma).reshape(n_frames,cn,-1)
                k_dilute_hidden_states = Dilution(k_hidden_states[n_frames:2 * n_frames], kasf, eg_mask, self.gamma).reshape(n_frames,cn,-1)
                q_dilute_hidden_states = torch.transpose(q_dilute_hidden_states, 1, 2)
                k_dilute_hidden_states = torch.transpose(k_dilute_hidden_states, 1, 2)

                # dilutional inject unconditional
                q[2 * n_frames:3 * n_frames] = q_dilute_hidden_states
                k[2 * n_frames:3 * n_frames] = k_dilute_hidden_states
                # dilutional inject conditional
                q[3 * n_frames:] = q_dilute_hidden_states
                k[3 * n_frames:] = k_dilute_hidden_states
            k_init = k[:n_frames]
            k_source = k[n_frames:2 * n_frames]
            k_uncond = k[2 * n_frames:3 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            k_cond = k[3 * n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

            v_init = v[:n_frames]
            v_source = v[n_frames:2 * n_frames]
            v_uncond = v[2 * n_frames:3 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_cond = v[3 * n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

            q_init = self.head_to_batch_dim(q[:n_frames])
            q_source = self.head_to_batch_dim(q[n_frames:2 * n_frames])
            q_uncond = self.head_to_batch_dim(q[2 * n_frames:3 * n_frames])
            q_cond = self.head_to_batch_dim(q[3 * n_frames:])

            k_init = self.head_to_batch_dim(k_init)
            k_source = self.head_to_batch_dim(k_source)
            k_uncond = self.head_to_batch_dim(k_uncond)
            k_cond = self.head_to_batch_dim(k_cond)

            v_init = self.head_to_batch_dim(v_init)
            v_source = self.head_to_batch_dim(v_source)
            v_uncond = self.head_to_batch_dim(v_uncond)
            v_cond = self.head_to_batch_dim(v_cond)

            q_init = q_init.view(n_frames, h, sequence_length, dim // h)
            k_init = k_init.view(n_frames, h, sequence_length, dim // h)
            v_init = v_init.view(n_frames, h, sequence_length, dim // h)

            q_src = q_source.view(n_frames, h, sequence_length, dim // h)
            k_src = k_source.view(n_frames, h, sequence_length, dim // h)
            v_src = v_source.view(n_frames, h, sequence_length, dim // h)

            q_uncond = q_uncond.view(n_frames, h, sequence_length, dim // h)
            k_uncond = k_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_uncond = v_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            
            q_cond = q_cond.view(n_frames, h, sequence_length, dim // h)
            k_cond = k_cond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_cond = v_cond.view(n_frames, h, sequence_length * n_frames, dim // h)

            out_init_all = []
            out_source_all = []
            out_uncond_all = []
            out_cond_all = []
            
            single_batch = n_frames<=12
            b = n_frames if single_batch else 1

            for frame in range(0, n_frames, b):
                out_init = []
                out_source = []
                out_uncond = []
                out_cond = []
                for j in range(h):
                    sim_init_b = torch.bmm(q_init[frame: frame+ b, j], k_init[frame: frame+ b, j].transpose(-1, -2)) * self.scale # [4, 4096, 4096]
                    sim_source_b = torch.bmm(q_src[frame: frame+ b, j], k_src[frame: frame+ b, j].transpose(-1, -2)) * self.scale # [4, 4096, 4096]
                    sim_uncond_b = torch.bmm(q_uncond[frame: frame+ b, j], k_uncond[frame: frame+ b, j].transpose(-1, -2)) * self.scale # [4, 4096, 16384]
                    sim_cond = torch.bmm(q_cond[frame: frame+ b, j], k_cond[frame: frame+ b, j].transpose(-1, -2)) * self.scale # [4, 4096, 16384]

                    out_init.append(torch.bmm(sim_init_b.softmax(dim=-1), v_init[frame: frame+ b, j]))
                    out_source.append(torch.bmm(sim_source_b.softmax(dim=-1), v_src[frame: frame+ b, j]))
                    out_uncond.append(torch.bmm(sim_uncond_b.softmax(dim=-1), v_uncond[frame: frame+ b, j]))
                    out_cond.append(torch.bmm(sim_cond.softmax(dim=-1), v_cond[frame: frame+ b, j]))
                
                out_init = torch.cat(out_init, dim=0)
                out_source = torch.cat(out_source, dim=0)
                out_uncond = torch.cat(out_uncond, dim=0) 
                out_cond = torch.cat(out_cond, dim=0) 
                if single_batch:
                    out_init = out_init.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                    out_source = out_source.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                    out_uncond = out_uncond.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                    out_cond = out_cond.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                out_init_all.append(out_init)
                out_source_all.append(out_source)
                out_uncond_all.append(out_uncond)
                out_cond_all.append(out_cond)
            
            out_init = torch.cat(out_init_all, dim=0)
            out_source = torch.cat(out_source_all, dim=0)
            out_uncond = torch.cat(out_uncond_all, dim=0)
            out_cond = torch.cat(out_cond_all, dim=0)
                
            out = torch.cat([out_init, out_source, out_uncond, out_cond], dim=0)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward
    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            # att1.forward = self attention on image patches
            module.attn1.forward = sa_forward(module.attn1)
            setattr(module.attn1, 'injection_schedule', [])
    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)

def propagation_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class propagation(block_class):

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
            low_freq=None,
        ) -> torch.Tensor:

            batch_size, sequence_length, dim = hidden_states.shape
            n_frames = batch_size // 4
            hidden_states = hidden_states.view(4, n_frames, sequence_length, dim)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states.view(4, n_frames, sequence_length, dim)

            if self.propagation_pass:
                self.keyframe_hidden_states = norm_hidden_states
            else:
                idx1 = []
                idx2 = []
                batch_idxs = [self.batch_idx]
                if self.batch_idx > 0:
                    batch_idxs.append(self.batch_idx - 1)
                sim = batch_cosine_sim(norm_hidden_states[1].reshape(-1, dim),
                                        self.keyframe_hidden_states[1][batch_idxs].reshape(-1, dim))
                if len(batch_idxs) == 2:
                    sim1, sim2 = sim.chunk(2, dim=1)
                    # sim: n_frames * seq_len, len(batch_idxs) * seq_len
                    idx1.append(sim1.argmax(dim=-1))  # n_frames * seq_len
                    idx2.append(sim2.argmax(dim=-1))  # n_frames * seq_len
                else:
                    idx1.append(sim.argmax(dim=-1))
                idx1 = torch.stack(idx1 * 4, dim=0)
                idx1 = idx1.squeeze(1)
                if len(batch_idxs) == 2:
                    idx2 = torch.stack(idx2 * 4, dim=0)
                    idx2 = idx2.squeeze(1)
            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if self.propagation_pass:
                self.attn_output = self.attn1(
                        norm_hidden_states.view(batch_size, sequence_length, dim),
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        **cross_attention_kwargs,
                    )
                self.kf_attn_output = self.attn_output
            else:
                batch_kf_size, _, _ = self.kf_attn_output.shape
                self.attn_output = self.kf_attn_output.view(4, batch_kf_size // 4, sequence_length, dim)[:,batch_idxs]
            if self.use_ada_layer_norm_zero:
                self.attn_output = gate_msa.unsqueeze(1) * self.attn_output

            # gather values from attn_output, using idx as indices, and get a tensor of shape 3, n_frames, seq_len, dim
            if not self.propagation_pass:
                if len(batch_idxs) == 2:
                    attn_1, attn_2 = self.attn_output[:, 0], self.attn_output[:, 1]
                    attn_output1 = attn_1.gather(dim=1, index=idx1.unsqueeze(-1).repeat(1, 1, dim))
                    attn_output2 = attn_2.gather(dim=1, index=idx2.unsqueeze(-1).repeat(1, 1, dim))
                    s = torch.arange(0, n_frames).to(idx1.device) + batch_idxs[0] * n_frames
                    # distance from the keyframe
                    p1 = batch_idxs[0] * n_frames + n_frames // 2
                    p2 = batch_idxs[1] * n_frames + n_frames // 2
                    d1 = torch.abs(s - p1)
                    d2 = torch.abs(s - p2)
                    # weight
                    w1 = d2 / (d1 + d2)
                    w1 = torch.sigmoid(w1)
                    w1 = w1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(4, 1, sequence_length, dim)
                    attn_output1 = attn_output1.view(4, n_frames, sequence_length, dim)
                    attn_output2 = attn_output2.view(4, n_frames, sequence_length, dim)
                    attn_output = w1 * attn_output1 + (1 - w1) * attn_output2
                else:
                    attn_output = self.attn_output[:,0].gather(dim=1, index=idx1.unsqueeze(-1).repeat(1, 1, dim))
                    attn_output = attn_output.view(4, n_frames, sequence_length, dim)
                attn_output = attn_output.reshape(batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
            else:
                attn_output = self.attn_output
            hidden_states = hidden_states.reshape(batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                attn_output = attn_output.reshape(batch_size, sequence_length, dim)
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            ff_output = self.ff(norm_hidden_states)
            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            hidden_states = ff_output + hidden_states
            return hidden_states
    return propagation


def causal_attention_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class causal_attention(block_class):
        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:

            batch_size, sequence_length, dim = hidden_states.shape
            n_frames = batch_size // 4
            hidden_states = hidden_states.view(4, n_frames, sequence_length, dim)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states.view(4, n_frames, sequence_length, dim)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            # Method of TAV and FateZero
            index = 'mid'
            # TAV
            if index == 'first':
                f = [0] * n_frames
                frame_index = torch.arange(n_frames) - 1
                frame_index[0] = 0
            # FateZero
            elif index == 'last':
                f = [clip_length-1] * n_frames
                frame_index = torch.arange(n_frames)
            elif index == 'mid':
                f = [int(n_frames-1)//2] * n_frames
                frame_index = torch.arange(n_frames)
            else:
                f = [0] * n_frames
                frame_index = torch.arange(n_frames)

            key = torch.cat([norm_hidden_states[:, f], norm_hidden_states[:, frame_index]], dim=2)
            query = torch.cat([norm_hidden_states, norm_hidden_states], dim=2)
            attn_output = self.attn1(
                    query.view(batch_size, sequence_length*2, dim),
                    encoder_hidden_states=key.view(batch_size, sequence_length*2, dim),
                    **cross_attention_kwargs,
                )
            attn_output = attn_output[:,:sequence_length,:]
            if self.use_ada_layer_norm_zero:
                self.attn_output = gate_msa.unsqueeze(1) * self.attn_output

            hidden_states = hidden_states.reshape(batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                attn_output = attn_output.reshape(batch_size, sequence_length, dim)
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            ff_output = self.ff(norm_hidden_states)
            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            hidden_states = ff_output + hidden_states
            return hidden_states
    return causal_attention

def basic_attention_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class basic_attention(block_class):
        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
            low_freq=None,
        ) -> torch.Tensor:

            batch_size, sequence_length, dim = hidden_states.shape
            n_frames = batch_size // 4
            hidden_states = hidden_states.view(4, n_frames, sequence_length, dim)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states.view(4, n_frames, sequence_length, dim)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            attn_output = self.attn1(
                    norm_hidden_states.view(batch_size, sequence_length, dim),
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    **cross_attention_kwargs,
                )
            if self.use_ada_layer_norm_zero:
                self.attn_output = gate_msa.unsqueeze(1) * self.attn_output

            hidden_states = hidden_states.reshape(batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                attn_output = attn_output.reshape(batch_size, sequence_length, dim)
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            ff_output = self.ff(norm_hidden_states)
            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            hidden_states = ff_output + hidden_states
            return hidden_states
    return basic_attention

def register_temporal_module(model: torch.nn.Module, module_name):

    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            if module_name == 'propagation':
                block_fn = propagation_block
            elif module_name == 'basic_attention':
                block_fn = basic_attention_block
            elif module_name == 'causal_attention':
                block_fn = causal_attention_block
            else:
                assert 0, 'No video quality enhancement module'

            module.__class__ = block_fn(module.__class__)

            # Something needed for older versions of diffusers
            if not hasattr(module, "use_ada_layer_norm_zero"):
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False

    return model

def make_editing_guidance_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class EditingGuidanceBlock(block_class):

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:

            if cross_attention_kwargs is not None:
                if cross_attention_kwargs.get("scale", None) is not None:
                    logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
    
            # Notice that normalization is always applied before the real computation in the following blocks.
            # 0. Self-Attention
            batch_size = hidden_states.shape[0]
    
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.norm_type == "ada_norm_zero":
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif self.norm_type == "ada_norm_single":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                norm_hidden_states = norm_hidden_states.squeeze(1)
            else:
                raise ValueError("Incorrect norm used")
            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)
    
            # 1. Prepare GLIGEN inputs
            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)
    
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.norm_type == "ada_norm_zero":
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.norm_type == "ada_norm_single":
                attn_output = gate_msa * attn_output
    
            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)
    
            # 1.2 GLIGEN Control
            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])
    
            # 3. Cross-Attention
            if self.attn2 is not None:
                if self.norm_type == "ada_norm":
                    norm_hidden_states = self.norm2(hidden_states, timestep)
                elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                    norm_hidden_states = self.norm2(hidden_states)
                elif self.norm_type == "ada_norm_single":
                    # For PixArt norm2 isn't applied here:
                    # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                    norm_hidden_states = hidden_states
                elif self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                else:
                    raise ValueError("Incorrect norm")
    
                if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                    norm_hidden_states = self.pos_embed(norm_hidden_states)
                
                if self.target_save and self.attn2.target_block and norm_hidden_states.shape[-1] == self.target_layer_dim:
                    query = self.attn2.to_q(norm_hidden_states)
                    key = self.attn2.to_k(encoder_hidden_states)
                    attmap = self.attn2.get_attention_scores(key, query, encoder_attention_mask)

                    #attmap = torch.einsum('bld,bnd->bln', query, key)
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
                            #img.save(self.edit_guidance_path+f'/{word}_{j}.jpg')
                            data = data + [np.array(img)]
                        data = np.stack(data, axis=0)
                        np.save(self.edit_guidance_path+f'/{word}', data)
                    print("editing guidance is saved")

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states
            # 4. Feed-forward
            # i2vgen doesn't have this norm 🤷<200d>♂️
            if self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif not self.norm_type == "ada_norm_single":
                norm_hidden_states = self.norm3(hidden_states)
    
            if self.norm_type == "ada_norm_zero":
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    
            if self.norm_type == "ada_norm_single":
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
    
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
            else:
                ff_output = self.ff(norm_hidden_states)
    
            if self.norm_type == "ada_norm_zero":
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif self.norm_type == "ada_norm_single":
                ff_output = gate_mlp * ff_output
    
            hidden_states = ff_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)
    
            return hidden_states

    return EditingGuidanceBlock

def set_editing_guidance(
        model: torch.nn.Module):

    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            target_fn = make_editing_guidance_block 
            module.__class__ = target_fn(module.__class__)

            # Something needed for older versions of diffusers
            if not hasattr(module, "use_ada_layer_norm_zero"):
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False

    return model
