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


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                        
                self.update_mask = tnf.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)


        if self.return_mask:
            return output, self.update_mask
        else:
            return output


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
        self.conv = nn.Conv2d(ed, ed, 3, 1, 1, groups=ed)
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


class BaseBlockv1(nn.Module): 
    default_embedding_dim = 256  
    def __init__(self,
                 in_dim, 
                 emb_dim=None,
                 out_dim=None,
                 shortcut=True):
        super().__init__()
        if in_dim == None:
            in_dim = self.default_embedding_dim
        if out_dim == None:
            out_dim = in_dim
        if emb_dim == None:
            emb_dim = out_dim
        self.pc1 = nn.Conv2d(in_dim, emb_dim, 1)
        self.act = WSiLU()
        self.mixer = RepVGGDW(emb_dim)
        self.pc2 = nn.Conv2d(emb_dim, emb_dim, 1)
        self.shortcut = shortcut
        self.ffn = FFN(emb_dim, int(emb_dim * 2))
        self.use_outlayer = False
        self.use_adaptor = False
        if emb_dim != out_dim:
            self.use_outlayer = True
            self.outlayer = nn.Conv2d(emb_dim, out_dim, 1)
        if in_dim != out_dim:
            self.use_adaptor = True
            self.adaptor = nn.Conv2d(in_dim, out_dim, 1)
        
    def forward(self, x):
        out = self.act(self.pc1(x))
        out = self.pc2(self.mixer(out))
        out = self.ffn(out)
        if self.use_outlayer:
            out = self.outlayer(out)
        if self.shortcut:
            if self.use_adaptor:
                out = out + self.adaptor(x)
            else:
                out = out + x

        return out


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
        x = self.layer_norm(x)
        scale, shift = self.embedding_layer(emb).chunk(2, dim=1)
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

