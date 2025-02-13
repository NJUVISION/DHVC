from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as tnf

from timm.layers.mlp import Mlp

from .entropy_coding import DiscretizedGaussian, gaussian_log_prob_mass
from .bound_ops import LowerBound, UpperBound


def get_conv(in_ch, out_ch, kernel_size, stride, padding, zero_bias=True, zero_weights=False):
    conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
    if zero_bias:
        conv.bias.data.mul_(0.0)
    if zero_weights:
        conv.weight.data.mul_(0.0)
    return conv


def conv_k1s1(in_ch, out_ch, zero_bias=True, zero_weights=False):
    return get_conv(in_ch, out_ch, 1, 1, 0, zero_bias, zero_weights)


def conv_k3s1(in_ch, out_ch, zero_bias=True, zero_weights=False):
    return get_conv(in_ch, out_ch, 3, 1, 1, zero_bias, zero_weights)


def conv_k5s1(in_ch, out_ch, zero_bias=True, zero_weights=False):
    return get_conv(in_ch, out_ch, 5, 1, 2, zero_bias, zero_weights)


def conv_k3s2(in_ch, out_ch):
    return get_conv(in_ch, out_ch, kernel_size=3, stride=2, padding=1)


def patch_downsample(in_ch, out_ch, rate=2):
    return get_conv(in_ch, out_ch, kernel_size=rate, stride=rate, padding=0)


def patch_upsample(in_ch, out_ch, rate=2):
    conv = nn.Sequential(
        get_conv(in_ch, out_ch * (rate ** 2), kernel_size=1, stride=1, padding=0),
        nn.PixelShuffle(rate)
    )
    return conv


def quantize_ste(x):
    """Differentiable quantization via the Straight-Through-Estimator."""
    return (torch.round(x) - x).detach() + x


class MyConvNeXtBlock(nn.Module):
    def __init__(self, dim, out_dim=None, kernel_size=7, mlp_ratio=2,
                 residual=True, ls_init_value=1e-6):
        super().__init__()
        # depthwise conv
        pad = (kernel_size - 1) // 2
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim)
        # layer norm
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.norm.affine = True
        # MLP
        hidden = int(mlp_ratio * dim)
        out_dim = out_dim or dim
        self.mlp = Mlp(dim, hidden_features=hidden, out_features=out_dim, act_layer=nn.GELU)
        # layer scaling
        if ls_init_value >= 0:
            self.gamma = nn.Parameter(torch.ones(1, out_dim, 1, 1) * float(ls_init_value))
        else:
            self.gamma = None
        self.residual = residual

    def forward(self, x):
        shortcut = x
        # depthwise conv
        x = self.conv_dw(x)
        # layer norm + MLP
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        # scaling
        if self.gamma is not None:
            x = x.mul(self.gamma)
        if self.residual:
            x = x + shortcut
        return x


def myconvnext_down(dim, new_dim, kernel_size=7):
    module = nn.Sequential(
        MyConvNeXtBlock(dim, kernel_size=kernel_size),
        conv_k3s2(dim, new_dim),
    )
    return module


class FeatureExtractor(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.enc_blocks = nn.ModuleList(blocks)

    def forward(self, x):
        feature = x
        enc_features = OrderedDict()
        for i, block in enumerate(self.enc_blocks):
            feature = block(feature)
            enc_features[int(feature.shape[2])] = feature
        return enc_features


class TemporalContextPrediction(nn.Module):
    def __init__(self, width, kernel_size=7):
        super().__init__()

        N_T = 4
        hidden_dim = 128

        enc_layers1 = [conv_k3s1(width*2, hidden_dim)]
        for i in range(1, N_T-1):
            enc_layers1.append(MyConvNeXtBlock(hidden_dim, kernel_size=kernel_size))
        enc_layers1.append(MyConvNeXtBlock(hidden_dim, kernel_size=kernel_size))

        dec_layers1 = [MyConvNeXtBlock(hidden_dim, kernel_size=kernel_size)]
        for _ in range(1, N_T-1):
            dec_layers1.append(
                nn.Sequential(
                    MyConvNeXtBlock(hidden_dim*2, kernel_size=kernel_size), 
                    conv_k3s1(hidden_dim*2, hidden_dim)),
                    )
        dec_layers1.append(conv_k3s1(hidden_dim*2, width))

        enc_layers2 = [conv_k3s1(width*2, hidden_dim)]
        for i in range(1, N_T-1):
            enc_layers2.append(MyConvNeXtBlock(hidden_dim, kernel_size=kernel_size))
        enc_layers2.append(MyConvNeXtBlock(hidden_dim, kernel_size=kernel_size))

        dec_layers2 = [MyConvNeXtBlock(hidden_dim, kernel_size=kernel_size)]
        for _ in range(1, N_T-1):
            dec_layers2.append(
                nn.Sequential(
                    MyConvNeXtBlock(hidden_dim*2, kernel_size=kernel_size), 
                    conv_k3s1(hidden_dim*2, hidden_dim)),
                    )
        dec_layers2.append(conv_k3s1(hidden_dim*2, width))

        self.enc1 = nn.Sequential(*enc_layers1)
        self.dec1 = nn.Sequential(*dec_layers1)

        self.enc2 = nn.Sequential(*enc_layers2)
        self.dec2 = nn.Sequential(*dec_layers2)

        self.end_fuse = MyConvNeXtBlock(width, kernel_size=kernel_size)

        self.N_T = N_T

    def forward(self, feature, ctx_feature1, ctx_feature2):
        zs = torch.cat([ctx_feature1, ctx_feature2], dim=1)
        # encoder
        skips = []
        z = zs
        for i in range(self.N_T):
            z = self.enc1[i](z)
            if i < self.N_T - 1:
                skips.append(z)
        # decoder
        z = self.dec1[0](z)
        for i in range(1, self.N_T):
            z = self.dec1[i](torch.cat([z, skips[-i]], dim=1))
        ctx_feature = z

        zs = torch.cat([feature, ctx_feature], dim=1)
        # encoder
        skips = []
        z = zs
        for i in range(self.N_T):
            z = self.enc2[i](z)
            if i < self.N_T - 1:
                skips.append(z)
        # decoder
        z = self.dec2[0](z)
        for i in range(1, self.N_T):
            z = self.dec2[i](torch.cat([z, skips[-i]], dim=1))
        ctx_feature = z

        ctx_feature = self.end_fuse(ctx_feature)

        return ctx_feature


class TemporalLatentBlock(nn.Module):
    def __init__(self, width, zdim, enc_width=None, kernel_size=7, mlp_ratio=2):
        super().__init__()
        self.in_channels  = width
        self.out_channels = width

        enc_width = enc_width or width
        concat_ch = (width * 2) if (enc_width is None) else (width + enc_width)
        self.resnet_front   = MyConvNeXtBlock(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.resnet_end     = MyConvNeXtBlock(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.resnet_dec     = MyConvNeXtBlock(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.posterior0 = MyConvNeXtBlock(enc_width, kernel_size=kernel_size)
        self.posterior1 = MyConvNeXtBlock(width, kernel_size=kernel_size)
        self.posterior2 = MyConvNeXtBlock(width, kernel_size=kernel_size)
        self.post_merge  = conv_k1s1(concat_ch, width)
        self.dec_merge   = conv_k1s1(width * 3, width)
        self.posterior   = conv_k3s1(width, zdim)
        self.z_proj      = conv_k1s1(zdim, width)
        self.prior       = conv_k1s1(width, zdim*2)

        self.tcp = TemporalContextPrediction(width=width, kernel_size=kernel_size)

        self.up_bound = UpperBound(10)

        self.discrete_gaussian = DiscretizedGaussian()
        self.is_latent_block = True
    
    def transform_prior(self, feature, ctx_feature):
        """ prior p(z_i | z_<i)

        Args:
            feature (torch.Tensor): feature map
        """
        feature = self.resnet_front(feature)
        ctx_feature = self.tcp(feature, ctx_feature[0], ctx_feature[1])
        pm, plogv = self.prior(ctx_feature).chunk(2, dim=1)
        plogv = tnf.softplus(plogv + 2.3) - 2.3 # make logscale > -2.3
        plogv = self.up_bound(plogv)
        pv = torch.exp(plogv)
        return feature, ctx_feature, pm, pv

    def transform_posterior(self, feature, enc_feature):
        """ posterior q(z_i | z_<i, x)

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        assert feature.shape[2:4] == enc_feature.shape[2:4]
        enc_feature = self.posterior0(enc_feature)
        feature = self.posterior1(feature)
        merged = torch.cat([feature, enc_feature], dim=1)
        merged = self.post_merge(merged)
        merged = self.posterior2(merged)
        qm = self.posterior(merged)
        return qm

    def fuse_feature_and_z(self, feature, z):
        # add the new information carried by z to the feature
        feature = feature + self.z_proj(z)
        return feature
    
    def fuse_dec_feature(self, feature, dec_feature, ctx_feature):
        if dec_feature is not None:
            dec_feature = torch.cat([feature, dec_feature, ctx_feature], dim=1)
            dec_feature = self.dec_merge(dec_feature)
            dec_feature = self.resnet_dec(dec_feature)
        else:
            dec_feature = torch.zeros_like(feature)
            dec_feature = torch.cat([feature, dec_feature, ctx_feature], dim=1)
            dec_feature = self.dec_merge(dec_feature)
            dec_feature = self.resnet_dec(dec_feature)
        return dec_feature

    def forward(self, feature, dec_feature, ctx_feature=None, enc_feature=None, mode='trainval',
                get_latent=False, strings=None):
        """ a complicated forward function

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        feature, ctx_feature, pm, pv = self.transform_prior(feature, ctx_feature)

        additional = dict()
        if mode == 'trainval': # training or validation
            qm = self.transform_posterior(feature, enc_feature)
            if self.training: # if training, use additive uniform noise
                z = quantize_ste(qm - pm) + pm
                z_uni = qm + torch.empty_like(qm).uniform_(-0.5, 0.5)
                log_prob = gaussian_log_prob_mass(pm, pv, x=z_uni, bin_size=1.0, prob_clamp=1e-6)
                kl = -1.0 * log_prob
            else: # if evaluation, use residual quantization
                z, probs = self.discrete_gaussian(qm, scales=pv, means=pm)
                kl = -1.0 * torch.log(probs)
            additional['kl'] = kl
        elif mode == 'progressive':
            z = pm
            additional['kl'] = torch.zeros_like(z)
        elif mode == 'compress': # encode z into bits
            qm = self.transform_posterior(feature, enc_feature)
            indexes = self.discrete_gaussian.build_indexes(pv)
            strings = self.discrete_gaussian.compress(qm, indexes, means=pm)
            z = self.discrete_gaussian.quantize(qm, mode='dequantize', means=pm)
            additional['strings'] = strings
        elif mode == 'decompress': # decode z from bits
            assert strings is not None
            indexes = self.discrete_gaussian.build_indexes(pv)
            z = self.discrete_gaussian.decompress(strings, indexes, means=pm)
        else:
            raise ValueError(f'Unknown mode={mode}')

        feature = self.fuse_feature_and_z(feature, z)
        feature = self.resnet_end(feature)

        dec_feature = self.fuse_dec_feature(feature, dec_feature, ctx_feature)
        
        if get_latent:
            additional['z'] = feature

        return feature, dec_feature, additional

    def update(self):
        self.discrete_gaussian.update()


