from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as tnf
from torch.nn.modules.utils import _pair

class WConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, groups=1, dilation=1, bias=False, den=None):
        super().__init__()
        if den is None:
            if kernel_size == 3:
                den = [0.75]
            elif kernel_size == 5:
                den = [1.5, 0.75]
            elif kernel_size == 7:
                den = [3.0, 1.5, 0.75]
            else:
                raise ValueError(f"Unsupported kernel size: {kernel_size}")
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.kernel_size = _pair(kernel_size)
        self.groups = groups
        self.dilation = _pair(dilation)
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        alfa = torch.tensor(den + [1.0] + den[::-1])
        self.register_buffer("alfa", alfa)
        self.register_buffer("Phi", torch.outer(alfa, alfa))
        if self.Phi.shape != self.kernel_size:
            raise ValueError(
                f"Phi shape {self.Phi.shape} must match {self.kernel_size}")

    def forward(self, x):
        weight = self.weight * self.Phi.to(x.device)
        padding = self.padding
        if getattr(self, "padding_mode", "zeros") == "replicate" and any(padding):
            x = tnf.pad(
                x, (padding[1], padding[1], padding[0], padding[0]),
                mode="replicate")
            padding = (0, 0)
        return tnf.conv2d(
            x, weight, bias=self.bias, stride=self.stride, padding=padding,
            groups=self.groups, dilation=self.dilation)


class WSiLU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(4.0 * x) * x


class WSiLUChunkAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = WSiLU()

    def forward(self, x):
        x1, x2 = self.silu(x).chunk(2, 1)
        return x1 + x2


def get_conv(in_ch, out_ch, kernel_size, stride, padding):
    if kernel_size == 1 or kernel_size % 2 == 0:
        return nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
    return WConv2d(in_ch, out_ch, kernel_size, stride, padding)


def conv_k1s1(in_ch, out_ch):
    return get_conv(in_ch, out_ch, 1, 1, 0)


def patch_downsample(in_ch, out_ch, rate=2):
    return get_conv(in_ch, out_ch, rate, rate, 0)


def patch_upsample(in_ch, out_ch, rate=2):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * rate ** 2, 1),
        nn.PixelShuffle(rate),
    )


class SetKey(nn.Module):
    def __init__(self, key):
        super().__init__()
        self.key = key

class FFN(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.pw1 = nn.Conv2d(channels, 2 * hidden, 1)
        self.act = WSiLUChunkAdd()
        self.pw2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        return x + self.pw2(self.act(self.pw1(x)))


def upsample_refine(channels):
    return FFN(channels, 2 * channels)


class RepVGGDW(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, groups=channels)

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x


class BaseBlockv1(nn.Module):
    def __init__(self, in_dim, emb_dim=None, out_dim=None, shortcut=True):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        emb_dim = out_dim if emb_dim is None else emb_dim
        self.pc1 = nn.Conv2d(in_dim, emb_dim, 1)
        self.act = WSiLU()
        self.mixer = RepVGGDW(emb_dim)
        self.pc2 = nn.Conv2d(emb_dim, emb_dim, 1)
        self.shortcut = shortcut
        self.ffn = FFN(emb_dim, 2 * emb_dim)
        self.use_outlayer = emb_dim != out_dim
        self.use_projection = in_dim != out_dim
        if self.use_outlayer:
            self.outlayer = nn.Conv2d(emb_dim, out_dim, 1)
        if self.use_projection:
            self.shortcut_projection = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x):
        out = self.act(self.pc1(x))
        out = self.pc2(self.mixer(out))
        out = self.ffn(out)
        if self.use_outlayer:
            out = self.outlayer(out)
        if self.shortcut:
            out = out + (
                self.shortcut_projection(x) if self.use_projection else x)
        return out


class FeatureExtractorWithEmbedding(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.enc_blocks = nn.ModuleList(blocks)

    def forward(self, x):
        features = OrderedDict()
        for index, block in enumerate(self.enc_blocks):
            if isinstance(block, SetKey):
                features[block.key] = x
            else:
                x = block(x)
        return features
