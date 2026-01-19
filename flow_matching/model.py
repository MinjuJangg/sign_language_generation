import math

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import trunc_normal_

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rel_pos_bias = RelativePositionBias(
            window_size=[int(num_patches**0.5), int(num_patches**0.5)], num_heads=num_heads)

    def forward(self, x, ids_keep=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        rp_bias = self.rel_pos_bias()
        attn += rp_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (
            2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(
                size=(window_size[0] * window_size[1],) * 2, dtype=relative_coords.dtype)
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index=relative_position_index

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1) 
        
        return relative_position_bias.permute(2, 0, 1).contiguous()

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        if t.dim() == 0:
            t = t[None]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    
class DiTBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    def __init__(self,
        img_resolution=32,
        patch_size=2,
        in_channels=4,
        out_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(img_resolution, patch_size, in_channels, hidden_size, bias=True)

        self.cond_embedder = PatchEmbed(img_resolution, patch_size, 4, hidden_size, bias=True)

        self.t_embedder = TimestepEmbedder(hidden_size)
        x_num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, x_num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,num_patches=x_num_patches) for _ in range(depth)])

        cond_num_patches=self.cond_embedder.num_patches
        self.cond_pos_embed=nn.Parameter(torch.zeros(1, cond_num_patches, hidden_size), requires_grad=False)

        self.mlp_dino_pose = nn.Sequential(
            nn.Linear(in_features=768, out_features=hidden_size, bias=True),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12)
        )
        self.mlp_dino_img = nn.Sequential(
            nn.Linear(in_features=1536, out_features=hidden_size, bias=True),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12)
        )
        self.conv1d_pose = nn.Sequential(
            nn.Conv1d(in_channels=257, out_channels=1, kernel_size=1),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12)
        )
        self.conv1d_global = nn.Sequential(
            nn.Conv1d(in_channels=257, out_channels=1, kernel_size=1),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12)
        )
        self.conv1d_local = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12)
        )
        self.conv1d_local2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12)
        )
        self.mlp_final = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12)
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        cond_pos_embed = get_2d_sincos_pos_embed(self.cond_pos_embed.shape[-1], int(self.cond_embedder.num_patches ** 0.5))
        self.cond_pos_embed.data.copy_(torch.from_numpy(cond_pos_embed).float().unsqueeze(0))

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        w2 = self.cond_embedder.proj.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.cond_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        nn.init.normal_(self.mlp_dino_pose[0].weight, std=0.02)
        nn.init.normal_(self.mlp_final[0].weight, std=0.02)
        nn.init.normal_(self.mlp_dino_img[0].weight, std=0.02)

        nn.init.normal_(self.conv1d_pose[0].weight, std=0.02)
        nn.init.normal_(self.conv1d_local[0].weight, std=0.02)
        nn.init.normal_(self.conv1d_global[0].weight, std=0.02)
        nn.init.normal_(self.conv1d_local2[0].weight, std=0.02)


    def unpatchify(self, x):

        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, t, x,dino_tgt_pose=None, src_img=None, dino_src_img=None, target_pose=None, prob=0,**kwargs):
        x = self.x_embedder(x) + self.pos_embed 
        t = self.t_embedder(t)  

        y2 = self.mlp_dino_pose(dino_tgt_pose)
        y2 = self.conv1d_pose(y2)

        y3 = self.mlp_dino_img(dino_src_img)
        y3 = self.conv1d_global(y3)

        src_img = self.cond_embedder(src_img) + self.cond_pos_embed
        y4 = self.conv1d_local(src_img)

        target_pose=self.cond_embedder(target_pose)+self.cond_pos_embed
        y5=self.conv1d_local2(target_pose)

        y = (y2 + y3 + y4 + y5).squeeze()
        y = self.mlp_final(y)

        if prob == 0:
            c = t + y  
        else:
            c = t 

        for block in self.blocks:
            x = block(x, c) 
        x = self.final_layer(x, c) 
        x = self.unpatchify(x) 
        return x

    def forward_with_cfg(self, t, x, dino_tgt_pose=None, src_img=None, dino_src_img=None, target_pose=None, cfg_scale=1.0,**kwargs):

        uncond_eps = self.forward(t,x, dino_tgt_pose, src_img,dino_src_img, target_pose,1)
        cond_eps = self.forward(t,x, dino_tgt_pose, src_img,dino_src_img, target_pose,0)

        eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

        return eps

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  

    emb = np.concatenate([emb_h, emb_w], axis=1)  
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  

    pos = pos.reshape(-1)  
    out = np.einsum("m,d->md", pos, omega) 

    emb_sin = np.sin(out)  
    emb_cos = np.cos(out)  

    emb = np.concatenate([emb_sin, emb_cos], axis=1) 
    return emb
def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-XL/8": DiT_XL_8,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-L/8": DiT_L_8,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-B/8": DiT_B_8,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
    "DiT-S/8": DiT_S_8,
}