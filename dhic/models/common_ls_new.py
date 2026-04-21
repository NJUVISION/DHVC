from collections import OrderedDict
import math
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as tnf
from torch.nn.modules.utils import _pair, _triple

from timm.layers.mlp import Mlp
from timm.models.layers import SqueezeExcite
from .ska import SKA

# ===========================================================================
# Basic
# ===========================================================================

class WConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1, dilation=1, bias=False, den=None):
        super(WConv2d, self).__init__()
        if den == None:
            if kernel_size == 3:
                den = [0.75]
            elif kernel_size == 5:
                den = [1.5, 0.75]
            elif kernel_size == 7:
                den = [3.0, 1.5, 0.75]
            else:
                print(f"kernel size {kernel_size} not implemented")       
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.kernel_size = _pair(kernel_size)
        self.groups = groups
        self.dilation = _pair(dilation)      
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')        
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        device = torch.device('cpu')  
        self.register_buffer('alfa', torch.cat([torch.tensor(den, device=device),torch.tensor([1.0], device=device),torch.flip(torch.tensor(den, device=device), dims=[0])]))
        self.register_buffer('Phi', torch.outer(self.alfa, self.alfa))

        if self.Phi.shape != self.kernel_size:
            raise ValueError(f"Phi shape {self.Phi.shape} must match kernel size {self.kernel_size}")

    def forward(self, x):
        Phi = self.Phi.to(x.device)
        weight_Phi = self.weight * Phi
        return tnf.conv2d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)
    

class WSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(4.0 * x) * x


class WSiLUChunkAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = WSiLU()

    def forward(self, x):
        x1, x2 = self.silu(x).chunk(2, 1)
        return x1 + x2

def get_conv(in_ch, out_ch, kernel_size, stride, padding, zero_bias=False, zero_weights=False):
    if kernel_size == 1 or kernel_size % 2 == 0:
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
    else:
        conv = WConv2d(in_ch, out_ch, kernel_size, stride, padding)
    if zero_bias:
        conv.bias.data.mul_(0.0)
    if zero_weights:
        conv.weight.data.mul_(0.0)
    return conv


def conv_k1s1(in_ch, out_ch, zero_bias=False, zero_weights=False):
    return get_conv(in_ch, out_ch, 1, 1, 0, zero_bias, zero_weights)

def conv_k3s1(in_ch, out_ch, zero_bias=False, zero_weights=False):
    return get_conv(in_ch, out_ch, 3, 1, 1, zero_bias, zero_weights)

def patch_downsample(in_ch, out_ch, rate=2):
    return get_conv(in_ch, out_ch, kernel_size=rate, stride=rate, padding=0)

def patch_upsample(in_ch, out_ch, rate=2):
    conv = nn.Sequential(
        nn.Conv2d(in_ch, out_ch * (rate ** 2), 1, 1, 0),
        nn.PixelShuffle(rate),
    )
    return conv

def mlp(dims):
    return nn.Sequential(
        nn.Linear(dims[0], dims[1]),
        WSiLU(),
        nn.Linear(dims[1], dims[1]),
    )


def quantize_ste(x):
    """Differentiable quantization via the Straight-Through-Estimator."""
    # STE (straight-through estimator) trick: x_hard - x_soft.detach() + x_soft
    return (torch.round(x) - x).detach() + x


class SetKey(nn.Module):
    """ A dummy layer that is used to mark the position of a layer in the network.
    """
    def __init__(self, key):
        super().__init__()
        self.key = key

    def forward(self, x):
        return x


class SetEmResidual(nn.Module):
    """ A dummy layer that is used to mark the position of a layer in the network.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.requires_dict_input = True
        self.adaptor = patch_upsample(in_ch, out_ch)

    def forward(self, fdict, x):
        x = x + self.adaptor(fdict["em_residual"])
        fdict["em_residual"] = x
        return x


class SetDecResidual(nn.Module):
    """ A dummy layer that is used to mark the position of a layer in the network.
    """
    def __init__(self):
        super().__init__()
        self.requires_dict_input = True
        self.adaptor = patch_upsample(in_ch, out_ch)

    def forward(self, fdict, x):
        x = x + self.adaptor(fdict["dec_residual"])
        fdict["dec_residual"] = x
        return x


class CompresionStopFlag(nn.Module):
    """ A dummy layer that is used to mark the stop position of encoding bits.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
# ===========================================================================
# LSBlock
# ===========================================================================

class FFN(nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = nn.Conv2d(ed, 2*h, 1, 1, 0)
        self.act = WSiLUChunkAdd()
        self.pw2 = nn.Conv2d(h, ed, 1, 1, 0)

    def forward(self, x):
        x = x + self.pw2(self.act(self.pw1(x)))
        return x
    
def upsample_refine(ch):
    return FFN(ch, 2*ch)

class RepVGGDW(nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = WConv2d(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
    
    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x
    
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = tnf.pad(conv1_w, [1,1,1,1])
        identity = tnf.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv

class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = nn.Conv2d(dim, dim // 2, 1, 1, 0)
        self.act = WSiLU()
        if lks > 1 and lks % 2 == 0:
            self.cv2 = WConv2d(dim // 2, dim // 2, kernel_size=lks, padding=(lks - 1) // 2, groups=dim // 2)
        else:
            self.cv2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=lks, padding=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = nn.Conv2d(dim // 2, dim // 2, 1, 1, 0)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        # self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        
        self.sks = sks
        self.groups = groups
        self.dim = dim
        
    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        # todo remove groupnorm
        # w = self.norm(self.cv4(x))
        w = self.cv4(x)
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w

class LSConv(nn.Module):
    def __init__(self, dim, lks=7, sks=3, groups=8):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=lks, sks=sks, groups=groups)
        self.ska = SKA()

    def forward(self, x):
        return self.ska(x, self.lkp(x)) + x


class EfficientLocalBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, dropout=0.1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.pwconv = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        x_res = x
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.dropout(x)
        x_ = x.flatten(2).transpose(1, 2)  # B, N, C
        x_ = self.norm(x_)
        x = x_.transpose(1, 2).view(B, C, H, W)
        return x + x_res


class RoPEGlobalAttention(nn.Module):
    """Rotary Embedding"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        N = H * W
        x_ = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x_ = self.norm(x_)

        qkv = self.qkv(x_).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # q/k/v: [B, heads, N, head_dim]

        # Apply RoPE (rotary position encoding)
        freqs = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, device=x.device) / self.head_dim))
        pos = torch.arange(N, device=x.device)
        sinusoid = torch.einsum("i,j->ij", pos, freqs)  # [N, dim/2]
        sin, cos = sinusoid.sin(), sinusoid.cos()
        sin, cos = [t.repeat_interleave(2, dim=-1)[None, None, :, :] for t in (sin, cos)]

        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        def apply_rope(x):
            return (x * cos) + (rotate_half(x) * sin)

        q = apply_rope(q)
        k = apply_rope(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        out = attn @ v  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        out = self.dropout(out)
        out = out.transpose(1, 2).view(B, C, H, W)
        return out


class HybridAttentionBlock(nn.Module):
    """Local Block + Global RoPE Attention"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.local = EfficientLocalBlock(dim, dropout=dropout)
        self.global_attn = RoPEGlobalAttention(dim, num_heads=num_heads, dropout=dropout)

    def forward(self, x):
        x_local = self.local(x)
        x_global = self.global_attn(x_local)
        return x_local + x_global


class LSBlock_Even(nn.Module):   
    default_embedding_dim = 256 
    def __init__(self,
                 ed):
        super().__init__()

        ed = ed or self.default_embedding_dim
            
        self.mixer = RepVGGDW(ed)
        # self.se = SqueezeExcite(ed, 0.25)
        self.ffn = FFN(ed, int(ed * 2))

    def forward(self, x):
        return self.ffn(self.mixer(x))
    
class LSBlock_Odd(nn.Module): 
    default_embedding_dim = 256   
    def __init__(self,
                 ed, 
                 lks=7,
                 sks=3,
                 groups=8,
                 kd=16,
                 use_atten=False):
        super().__init__()

        ed = ed or self.default_embedding_dim
            
        if use_atten:
            # todo attention
            # self.mixer = LightweightAttention(ed, key_dim=kd)
            self.mixer = HybridAttentionBlock(ed)
        else:
            self.mixer = LSConv(ed, lks=lks, sks=sks, groups=groups)

        self.ffn = FFN(ed, int(ed * 2))

    def forward(self, x):
        return self.ffn(self.mixer(x))

# ===========================================================================
# DC BLOCK
# ===========================================================================

class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut=False, force_adaptor=False):
        super().__init__()
        self.adaptor = None
        if in_ch != out_ch or force_adaptor:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)
        self.shortcut = shortcut
        self.dc = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            WSiLU(),
            WConv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            nn.Conv2d(out_ch, out_ch, 1),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch * 4, 1),
            WSiLUChunkAdd(),
            nn.Conv2d(out_ch * 2, out_ch, 1),
        )

        self.proxy = None

    def forward(self, x, quant_step=None, to_cat=None, cat_at_front=True):
        if self.adaptor is not None:
            x = self.adaptor(x)
        out = self.dc(x) + x
        out = self.ffn(out) + out
        if self.shortcut:
            out = out + x
        if quant_step is not None:
            out = out * quant_step
        if to_cat is not None:
            if cat_at_front:
                out = torch.cat((to_cat, out), dim=1)
            else:
                out = torch.cat((out, to_cat), dim=1)
        return out

# ===========================================================================
# ORI
# ===========================================================================

class FeatureExtractorWithEmbedding(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.enc_blocks = nn.ModuleList(blocks)

    def forward(self, x, emb=None):
        features = OrderedDict()
        for i, block in enumerate(self.enc_blocks):
            if isinstance(block, SetKey):
                features[block.key] = x
            elif getattr(block, 'requires_embedding', False):
                x = block(x, emb)
            else:
                x = block(x)
        return features
    
    def __getitem__(self, index):
        return self.enc_blocks[index]

    def __len__(self):
        return len(self.enc_blocks)


def sinusoidal_embedding(values: torch.Tensor, dim=256, max_period=64):
    assert values.dim() == 1 and (dim % 2) == 0
    exponents = torch.linspace(0, 1, steps=(dim // 2))
    freqs = torch.pow(max_period, -1.0 * exponents).to(device=values.device)
    args = values.view(-1, 1) * freqs.view(1, dim//2)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class Permute(nn.Module):
    def __init__(self, *dims: tuple):
        """ Permute dimensions of a tensor
        """
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, dims=self.dims)


class AdaptiveLayerNorm(nn.Module):
    """ Channel-last LayerNorm with adaptive affine parameters that depend on the \
        input embedding.
    """
    def __init__(self, dim: int, embed_dim: int):
        super().__init__()
        self.dim = dim
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.embedding_layer = nn.Sequential(
            WSiLU(),
            nn.Linear(embed_dim, 2*dim),
        )
        # TODO: initialize the affine parameters such that the initial transform is identity
        # self.embedding_layer[1].weight.data.mul_(0.01)
        # self.embedding_layer[1].bias.data.fill_(0.0)

    def forward(self, x, emb):
        # x: (B, ..., dim), emb: (B, embed_dim)
        x = self.layer_norm(x)
        scale, shift = self.embedding_layer(emb).chunk(2, dim=1) # (B, dim) x 2
        # (B, dim) -> (B, ..., dim)
        scale = torch.unflatten(scale, dim=1, sizes=[1] * (x.dim() - 2) + [self.dim])
        shift = torch.unflatten(shift, dim=1, sizes=[1] * (x.dim() - 2) + [self.dim])
        x = x * (1 + scale) + shift
        return x


class ConvNeXtBlockAdaLN(nn.Module):
    default_embedding_dim = 256
    def __init__(self, dim, embed_dim=None, out_dim=None, kernel_size=7, mlp_ratio=2,
                 residual=True, ls_init_value=1e-6, requires_q_scale=False):
        super().__init__()
        # depthwise conv
        pad = (kernel_size - 1) // 2
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim)
        # layer norm
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm.affine = False # for FLOPs computing
        # AdaLN
        embed_dim = embed_dim or self.default_embedding_dim
        self.embedding_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(embed_dim, 2*dim),
            nn.Unflatten(1, unflattened_size=(1, 1, 2*dim))
        )
        # MLP
        hidden = int(mlp_ratio * dim)
        out_dim = out_dim or dim
        self.mlp = Mlp(dim, hidden_features=hidden, out_features=out_dim, act_layer=nn.GELU)
        # layer scaling
        if ls_init_value >= 0:
            self.gamma = nn.Parameter(torch.full(size=(1, out_dim, 1, 1), fill_value=1e-6))
        else:
            self.gamma = None

        self.residual = residual
        self.requires_embedding = True
        self.requires_q_scale = requires_q_scale

    def forward(self, x, emb):
        shortcut = x
        # depthwise conv
        x = self.conv_dw(x)
        # layer norm
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        # AdaLN
        embedding = self.embedding_layer(emb)
        shift, scale = torch.chunk(embedding, chunks=2, dim=-1)
        x = x * (1 + scale) + shift
        # MLP
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        # scaling
        if self.gamma is not None:
            x = x.mul(self.gamma)
        if self.residual:
            x = x + shortcut
        return x

