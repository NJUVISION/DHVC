from collections import OrderedDict
import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as tnf

import models.common_ls_v1 as cm
import models.entropy_coding as entropy_coding
from models.bound_ops import UpperBound

from torchvision import transforms
from torchvision.utils import save_image
import torchvision.utils as vutils
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def custom_deepcopy(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    elif isinstance(obj, dict):
        return {key: custom_deepcopy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [custom_deepcopy(item) for item in obj]
    elif isinstance(obj, OrderedDict):
        return OrderedDict((key, custom_deepcopy(value)) for key, value in obj.items())
    else:
        return obj

class LatentVariableBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discrete_gaussian = entropy_coding.DiscretizedGaussian()
        self.requires_dict_input = True

    def transform_prior(self, feature):
        raise NotImplementedError()

    def transform_posterior(self, feature, enc_feature):
        raise NotImplementedError()
    
    def update(self):
        self.discrete_gaussian.update()

    def forward(self, fdict, f_ctx, use_dispersive=False, layer_id=0):
        feature = fdict['feature']
        mode = fdict['mode']
        ctx_feature = f_ctx
        lmb_em = torch.exp(fdict[f'em{layer_id+1}_lmb'])
        ctx_feature_modulated = ctx_feature * lmb_em 
        pm, pv = self.transform_prior(ctx_feature_modulated)
        z_scale = torch.exp(fdict[f'z{layer_id+1}_scale'])
        pm_scaled = pm / z_scale
        pv_scaled = pv / (z_scale ** 2)
        # =========================================================================

        if mode == 'progressive':
            z = pm

        elif mode == 'trainval':
            enc_feature = fdict['all_features'][self.enc_key]
            lmb_enc = torch.exp(fdict[f'enc{layer_id+1}_lmb'])
            enc_feature_modulated = enc_feature * lmb_enc

            qm = self.transform_posterior(feature, enc_feature_modulated)
            qm_scaled = qm / z_scale
            
            if self.training:
                z_scaled_ste = cm.quantize_ste(qm_scaled - pm_scaled) + pm_scaled
                z_uni_scaled = qm_scaled + torch.empty_like(qm_scaled).uniform_(-0.5, 0.5)
                log_prob = entropy_coding.gaussian_log_prob_mass(pm_scaled, pv_scaled, x=z_uni_scaled, bin_size=1.0, prob_clamp=1e-6)
                kl = -1.0 * log_prob
                z_scaled = z_scaled_ste
                
                # todo add_noise
                # z = qm + torch.empty_like(qm).uniform_(-0.5, 0.5)
                # log_prob = entropy_coding.gaussian_log_prob_mass(pm, pv, x=z, bin_size=1.0, prob_clamp=1e-6)
                # kl = -1.0 * log_prob
            else:
                z_scaled, probs = self.discrete_gaussian(qm_scaled, scales=pv_scaled, means=pm_scaled)
                kl = -1.0 * torch.log(probs)
            fdict['kl_divs'].append(kl)
            z = z_scaled * z_scale
            
        elif mode == 'compress':
            self.update()
            enc_feature = fdict['all_features'][self.enc_key]
            lmb_enc = torch.exp(fdict[f'enc{layer_id+1}_lmb'])
            enc_feature_modulated = enc_feature * lmb_enc

            qm = self.transform_posterior(feature, enc_feature_modulated)
            
            qm_scaled = qm / z_scale
            indexes = self.discrete_gaussian.build_indexes(pv_scaled)
            strings = self.discrete_gaussian.compress(qm_scaled, indexes, means=pm_scaled)
            z_scaled = self.discrete_gaussian.quantize(qm_scaled, mode='dequantize', means=pm_scaled)
            fdict['bit_strings'].append(strings)
            z = z_scaled * z_scale
            
        elif mode == 'decompress':
            strings = fdict['bit_strings'].pop(0)
            indexes = self.discrete_gaussian.build_indexes(pv_scaled)
            z_scaled = self.discrete_gaussian.decompress(strings, indexes, means=pm_scaled)
            z = z_scaled * z_scale
        else:
            raise ValueError(f'Unknown mode={mode}')
        feature = feature + self.z_proj(z)

        fdict['feature'] = feature
        fdict['all_features'][f'{self.name}_z'] = z
        fdict['all_features'][f'{self.name}_out'] = feature
        fdict['all_features'][f'{self.name}_ctx'] = ctx_feature
        return fdict


class VRLVBlock(LatentVariableBlock):
    default_embedding_dim = 256
    # todo add last zdim
    def __init__(self, dim, zdim, enc_key, enc_dim, name=None, emb_dim=None, kernel_size=7):
        super().__init__()
        self.in_channels  = dim
        self.out_channels = dim
        self.enc_key = enc_key
        self.name = name

        emb_dim = emb_dim or self.default_embedding_dim
        # todo predictor
        self.enc_resblock = cm.BaseBlockv1(enc_dim)
        self.posterior = cm.BaseBlockv1(dim + enc_dim, dim, zdim)
        # todo predictor
        self.z_proj = cm.conv_k1s1(zdim, dim)
        self.prior  = cm.conv_k1s1(dim, zdim*2)
        self.up_bound = UpperBound(10)

    def transform_prior(self, feature):
        """ prior p(z_i | z_<i)

        Args:
            feature (torch.Tensor): feature map
        """
        pm, plogv = self.prior(feature).chunk(2, dim=1)
        
        plogv = tnf.softplus(plogv + 2.3) - 2.3 # make logscale > -2.3
        plogv = self.up_bound(plogv)
        pv = torch.exp(plogv)
        return pm, pv

    def transform_posterior(self, feature, enc_feature):
        """ posterior q(z_i | z_<i, x)

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        assert feature.shape[2:4] == enc_feature.shape[2:4]
        enc_feature = self.enc_resblock((enc_feature))
        merged = torch.cat([feature, enc_feature], dim=1)
        qm = self.posterior(merged)
        return qm

class MergeFromEM(nn.Module):
    def __init__(self, key, in_dim, out_dim):
        super().__init__()
        self.key = key
        self.requires_dict_input = True
        self.merge = cm.BaseBlockv1(in_dim)
        self.outlayer = cm.conv_k1s1(in_dim, out_dim)

    def forward(self, fdict, layer_id=0):
        feature = fdict['dec_feature']      
        f_em_out = fdict['all_features'][f'{self.key}_out']
        f_ctx = fdict['all_features'][f'{self.key}_ctx']
        lmb_dec = torch.exp(fdict[f'dec{layer_id+1}_lmb'])
        f_em_out_modulated = f_em_out * lmb_dec
        f_ctx_modulated = f_ctx * lmb_dec

        assert feature.shape[2:4] == f_em_out.shape[2:4] == f_ctx.shape[2:4]
        feature = self.outlayer(self.merge(torch.cat([feature, f_em_out_modulated, f_ctx_modulated], dim=1)))

        fdict['dec_feature'] = feature
        return fdict


class VariableRateLossyVAE(nn.Module):
    log2_e = math.log2(math.e)
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = cm.FeatureExtractorWithEmbedding(config.pop('enc_blocks'))
        self.em_blocks = nn.ModuleList(config.pop('em_blocks'))
        self.dec_blocks = nn.ModuleList(config.pop('dec_blocks'))
        width = self.em_blocks[2].in_channels
        self.em_bias = nn.Parameter(torch.zeros(1, width, 1, 1))
        self.dec_bias = nn.Parameter(torch.zeros(1, width, 1, 1))

        self.register_buffer('_dummy', torch.zeros(1), persistent=False)
        self._dummy: torch.Tensor
        self.max_stride = config['max_stride']

        self.z_dims = config['z_dims']

        self.q_num = 64
        self.enc1_lmb_embedding = nn.Parameter(torch.zeros((self.q_num, 512, 1, 1))) 
        self.enc2_lmb_embedding = nn.Parameter(torch.zeros((self.q_num, 512, 1, 1))) 
        self.enc3_lmb_embedding = nn.Parameter(torch.zeros((self.q_num, 512, 1, 1))) 
        self.enc4_lmb_embedding = nn.Parameter(torch.zeros((self.q_num, 384, 1, 1))) 
        self.em1_lmb_embedding = nn.Parameter(torch.zeros((self.q_num, 512, 1, 1)))
        self.em2_lmb_embedding = nn.Parameter(torch.zeros((self.q_num, 512, 1, 1)))
        self.em3_lmb_embedding = nn.Parameter(torch.zeros((self.q_num, 512, 1, 1)))
        self.em4_lmb_embedding = nn.Parameter(torch.zeros((self.q_num, 384, 1, 1)))
        self.dec1_lmb_embedding = nn.Parameter(torch.zeros((self.q_num, 512, 1, 1)))
        self.dec2_lmb_embedding = nn.Parameter(torch.zeros((self.q_num, 512, 1, 1)))
        self.dec3_lmb_embedding = nn.Parameter(torch.zeros((self.q_num, 512, 1, 1)))
        self.dec4_lmb_embedding = nn.Parameter(torch.zeros((self.q_num, 384, 1, 1)))

        self.z1_scale_embedding = nn.Parameter(torch.zeros((self.q_num, self.z_dims[0], 1, 1)))
        self.z2_scale_embedding = nn.Parameter(torch.zeros((self.q_num, self.z_dims[1], 1, 1)))
        self.z3_scale_embedding = nn.Parameter(torch.zeros((self.q_num, self.z_dims[2], 1, 1)))
        self.z4_scale_embedding = nn.Parameter(torch.zeros((self.q_num, self.z_dims[3], 1, 1)))


    def sample_lmb(self, cur_qp=None, min_lamda=256, max_lamda=8192, q_num=64, bs=1, train_fixed=False):
        if cur_qp is None:
            if train_fixed:
                cur_qp = [q_num - 1] * bs
            else:
                cur_qp = random.choices(list(range(q_num)), k=bs)
        elif isinstance(cur_qp, int):
            cur_qp = [cur_qp] * bs
        q_index = torch.tensor(cur_qp, dtype=torch.float32)
        log_min = torch.log(torch.tensor(min_lamda, dtype=torch.float32))
        log_max = torch.log(torch.tensor(max_lamda, dtype=torch.float32))
        log_lambda = log_min + (q_index / (q_num - 1)) * (log_max - log_min)
        lmb_tensor = torch.exp(log_lambda)
        cur_lamda = [lmb for lmb in lmb_tensor]
        layer_qp = cur_qp
            
        return layer_qp, cur_lamda


    def get_initial_fdict(self, layer_qp, bias_bhw, use_dispersive=False):
        fdict = dict()
        nB, nH, nW = bias_bhw

        enc1_lmb, enc2_lmb, enc3_lmb, enc4_lmb = [], [], [], []
        em1_lmb, em2_lmb, em3_lmb, em4_lmb = [], [], [], []
        dec1_lmb, dec2_lmb, dec3_lmb, dec4_lmb = [], [], [], []
        z1_scale, z2_scale, z3_scale, z4_scale = [], [], [], []

        for j in range(nB):
            qp = layer_qp[j]
            enc1_lmb.append(self.enc1_lmb_embedding[qp])
            enc2_lmb.append(self.enc2_lmb_embedding[qp])
            enc3_lmb.append(self.enc3_lmb_embedding[qp])
            enc4_lmb.append(self.enc4_lmb_embedding[qp])
            
            em1_lmb.append(self.em1_lmb_embedding[qp])
            em2_lmb.append(self.em2_lmb_embedding[qp])
            em3_lmb.append(self.em3_lmb_embedding[qp])
            em4_lmb.append(self.em4_lmb_embedding[qp])
            
            dec1_lmb.append(self.dec1_lmb_embedding[qp])
            dec2_lmb.append(self.dec2_lmb_embedding[qp])
            dec3_lmb.append(self.dec3_lmb_embedding[qp])
            dec4_lmb.append(self.dec4_lmb_embedding[qp])

            z1_scale.append(self.z1_scale_embedding[qp])
            z2_scale.append(self.z2_scale_embedding[qp])
            z3_scale.append(self.z3_scale_embedding[qp])
            z4_scale.append(self.z4_scale_embedding[qp])

        fdict["enc1_lmb"] = torch.stack(enc1_lmb, dim=0) 
        fdict["enc2_lmb"] = torch.stack(enc2_lmb, dim=0) 
        fdict["enc3_lmb"] = torch.stack(enc3_lmb, dim=0) 
        fdict["enc4_lmb"] = torch.stack(enc4_lmb, dim=0) 
        
        fdict["em1_lmb"] = torch.stack(em1_lmb, dim=0)
        fdict["em2_lmb"] = torch.stack(em2_lmb, dim=0)
        fdict["em3_lmb"] = torch.stack(em3_lmb, dim=0)
        fdict["em4_lmb"] = torch.stack(em4_lmb, dim=0)

        fdict["dec1_lmb"] = torch.stack(dec1_lmb, dim=0)
        fdict["dec2_lmb"] = torch.stack(dec2_lmb, dim=0)
        fdict["dec3_lmb"] = torch.stack(dec3_lmb, dim=0)
        fdict["dec4_lmb"] = torch.stack(dec4_lmb, dim=0)
        
        fdict["z1_scale"] = torch.stack(z1_scale, dim=0)
        fdict["z2_scale"] = torch.stack(z2_scale, dim=0)
        fdict["z3_scale"] = torch.stack(z3_scale, dim=0)
        fdict["z4_scale"] = torch.stack(z4_scale, dim=0)

        fdict['em_residual'] = self.em_bias
        fdict['dec_residual'] = self.dec_bias
        fdict['all_features'] = OrderedDict()
        fdict['dispersive_loss'] = OrderedDict()
        
        fdict['feature'] = self.em_bias.expand(nB, -1, nH, nW)
        fdict['dec_feature'] = self.dec_bias.expand(nB, -1, nH, nW)
        fdict['kl_divs'] = []
        fdict['bit_strings'] = []
        return fdict

    def forward_bottomup(self, im, layer_qp, use_dispersive=False):
        bias_bhw = (im.shape[0], im.shape[2]//self.max_stride, im.shape[3]//self.max_stride)
        fdict = self.get_initial_fdict(layer_qp, bias_bhw, use_dispersive=use_dispersive)
        fdict['all_features'] = self.encoder(im)
        return fdict, im

    def forward_em(self, fdict, mode='trainval', use_dispersive=False):
        fdict['mode'] = mode

        i = 0
        for _, block in enumerate(self.em_blocks):
            if getattr(block, 'requires_dict_input', False):
                fdict = block(fdict, fdict['feature'], use_dispersive=use_dispersive, layer_id=i)
                i += 1
            else:
                fdict['feature'] = block(fdict['feature'])

        return fdict

    def forward_topdown(self, fdict, mode='trainval', use_dispersive=False):
        fdict = self.forward_em(fdict, mode, use_dispersive=use_dispersive)

        layer_id = 0
        for _, block in enumerate(self.dec_blocks):
            if getattr(block, 'requires_dict_input', False):
                fdict = block(fdict, layer_id=layer_id)
                layer_id += 1
            else:
                fdict['dec_feature'] = block(fdict['dec_feature'])

        # todo scale-wise pop out, if use normal decoding, remove this comment
        fdict['x_hat'] = fdict.pop('dec_feature')
        return fdict


    def forward_em_scale_wise(self, fdict, mode='trainval', scale=0, use_dispersive=False):
        fdict['mode'] = mode

        i = 0
        scales_split = [0, 6, 15, 24, 28]
        for _, block in enumerate(self.em_blocks[scales_split[scale]: scales_split[scale+1]+1]):
            if getattr(block, 'requires_dict_input', False):
                fdict = block(fdict, fdict['feature'], use_dispersive=use_dispersive, layer_id=scale)
                i += 1
            else:
                fdict['feature'] = block(fdict['feature'])

        return fdict

    def forward_topdown_scale_wise(self, fdict, mode='trainval', scale=0, use_dispersive=False):
        scales_split = [0, 6, 15, 24, 35]
        for _, block in enumerate(self.dec_blocks[scales_split[scale]: scales_split[scale+1]+1]):
            if getattr(block, 'requires_dict_input', False):
                fdict = block(fdict, layer_id=scale)
            else:
                fdict['dec_feature'] = block(fdict['dec_feature'])
        
        # todo scale-wise pop out
        return fdict
    

    def forward(self, im, cur_qp=None, return_fdict=False, train_fixed=False, use_dispersive=False, kl_rate=1.0):
        B, imC, imH, imW = im.shape
        layer_qp, lmb_list = self.sample_lmb(cur_qp=cur_qp, train_fixed=train_fixed, bs=B)
        lmb = torch.stack(lmb_list).to(self._dummy.device, non_blocking=True)
        im = im.to(self._dummy.device)
        fdict, x = self.forward_bottomup(im, layer_qp, use_dispersive=use_dispersive)
        # todo one-shot or test
        fdict = self.forward_topdown(fdict, mode='trainval', use_dispersive=use_dispersive)

        # todo scale-wise-regu
        # scale_wise_losses = []
        # for n in range(4):
        #     temp_fdict = custom_deepcopy(fdict)
        #     fdict = self.forward_em_scale_wise(fdict, mode='trainval', scale=n, use_dispersive=False)
        #     fdict = self.forward_topdown_scale_wise(fdict, mode='trainval', scale=n, use_dispersive=False)
        #     if n > 0:
        #         with torch.no_grad():
        #             temp_fdict = self.forward_em_scale_wise(temp_fdict, mode='progressive', scale=n, use_dispersive=False)
        #             temp_fdict = self.forward_topdown_scale_wise(temp_fdict, mode='progressive', scale=n, use_dispersive=False)
        #             scale_wise_losses.append(self.wavelet_loss(fdict['dec_feature'], temp_fdict['dec_feature']))
        # scale_wise_regu = torch.stack(scale_wise_losses).sum(dim=0)
        # fdict['x_hat'] = fdict.pop('dec_feature')

        # todo scale-wise-progressive
        # scale_wise_progressive_mse = []
        # scale_wise_progressive_bpp = []
        # scale_wise_progressive_loss = []
        # for n in range(4):
        #     fdict = self.forward_em_scale_wise(fdict, mode='trainval', scale=n, use_dispersive=False)
        #     fdict = self.forward_topdown_scale_wise(fdict, mode='trainval', scale=n, use_dispersive=False)
        #     if n < 3:
        #         with torch.no_grad():
        #             temp_fdict = custom_deepcopy(fdict)
        #             for pn in range(n+1, 4):
        #                 temp_fdict = self.forward_em_scale_wise(temp_fdict, mode='progressive', scale=pn, use_dispersive=False)
        #                 temp_fdict = self.forward_topdown_scale_wise(temp_fdict, mode='progressive', scale=pn, use_dispersive=False)
        #             temp_fdict['x_hat'] = temp_fdict.pop('dec_feature')
        #             temp_mse = (0.8 ** (3-n)) * tnf.smooth_l1_loss(temp_fdict['x_hat'], x, reduction='none').mean(dim=(1,2,3))
        #             temp_kivs = [kl.sum(dim=(1, 2, 3)) for kl in temp_fdict['kl_divs']]
        #             temp_bpp = sum(temp_kivs) * self.log2_e / float(imH * imW)
        #             # temp_loss = temp_bpp + lmb * temp_mse
        #             scale_wise_progressive_mse.append(temp_mse)
        #             scale_wise_progressive_bpp.append(temp_bpp)
        #             # scale_wise_progressive_loss.append(temp_loss)
        #     else:
        #         fdict['x_hat'] = fdict.pop('dec_feature')
        #         x_hat, kl_divs = fdict['x_hat'], fdict['kl_divs']
        #         temp_kivs = [kl.sum(dim=(1, 2, 3)) for kl in fdict['kl_divs']]
        #         # todo temp_bpp and bpp
        #         temp_bpp = sum(temp_kivs) * self.log2_e / float(imH * imW)
        #         temp_mse = tnf.smooth_l1_loss(x_hat, x, reduction='none').mean(dim=(1,2,3))
        #         # temp_loss = temp_bpp + lmb * temp_mse
        #         scale_wise_progressive_mse.append(temp_mse)
        #         scale_wise_progressive_bpp.append(temp_bpp)
                # scale_wise_progressive_loss.append(temp_loss)

        # ================ Compute Loss ================
        # todo when not progressive scale-wise mode, remove this comment 441 - 461
        x_hat, kl_divs = fdict['x_hat'], fdict['kl_divs']
        # rate
        kl_divs = [kl.sum(dim=(1, 2, 3)) for kl in kl_divs]
        bpp = sum(kl_divs) * self.log2_e / float(imH * imW)

        # distortion
        mse = tnf.mse_loss(x_hat, x, reduction='none').mean(dim=(1,2,3))
        # todo hierarchical kl annealing
        loss = kl_rate * (bpp) + lmb * mse

        metrics = OrderedDict()
        metrics['loss'] = loss.mean(0)

        # ================ Logging ================
        # todo when not progressive scale-wise mode, remove this comment
        with torch.inference_mode():
            metrics['bpp'] = bpp.mean(0)
            metrics['bpp1'] = 0.0
            metrics['bpp2'] = 0.0
            metrics['bpp3'] = 0.0
            metrics['bpp4'] = 0.0
            metrics['bpp_hyper'] = 0.0
            metrics['mse'] = mse.mean(0)
            if use_dispersive:
                metrics['disp'] = 0.0

        if return_fdict:
            return metrics, fdict
        return metrics


def dhvc(lmb_range=(16, 8192), pretrained=False):
    cfg = dict()
    cfg['max_stride'] = 64
    cfg['lmb_embed_dim'] = (256, 256)
    ch = 128
    enc_dims = [192, ch*3, ch*4, ch*4, ch*4]

    im_channels = 3
    cfg['enc_blocks'] = [
        # todo pixelunshuffle downsample
        nn.PixelUnshuffle(8),
        *[cm.BaseBlockv1(enc_dims[0]) for _ in range(10)],
        cm.WConv2d(enc_dims[0], enc_dims[1], kernel_size=3),
        # 8x8
        *[cm.BaseBlockv1(enc_dims[1]) for _ in range(8)],
        cm.SetKey('enc_s8'),
        cm.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[cm.BaseBlockv1(enc_dims[2]) for _ in range(6)],
        cm.SetKey('enc_s16'),
        cm.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[cm.BaseBlockv1(enc_dims[3]) for _ in range(4)],
        cm.SetKey('enc_s32'),
        cm.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[cm.BaseBlockv1(enc_dims[4]) for _ in range(2)],
        cm.SetKey('enc_s64'),
    ]

    dec_dims = [ch*4, ch*4, ch*4, ch*3, 192]
    z_dims = [320, 256, 128, 64]
    
    cfg['em_blocks'] = [
        # 1x1
        *[cm.BaseBlockv1(dec_dims[0]) for _ in range(2)],
        VRLVBlock(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_dim=enc_dims[-1], name='z1', kernel_size=1),
        *[cm.BaseBlockv1(dec_dims[0]) for _ in range(2)],
        cm.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        cm.upsample_refine(dec_dims[1]), 
        # 2x2
        *[cm.BaseBlockv1(dec_dims[1]) for _ in range(2)],
        VRLVBlock(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_dim=enc_dims[-2], name='z2', kernel_size=3),
        *[cm.BaseBlockv1(dec_dims[1]) for _ in range(2)],
        cm.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        cm.upsample_refine(dec_dims[2]), 
        # 4x4
        *[cm.BaseBlockv1(dec_dims[2]) for _ in range(3)],
        VRLVBlock(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_dim=enc_dims[-3], name='z3', kernel_size=5),
        *[cm.BaseBlockv1(dec_dims[2]) for _ in range(3)],
        cm.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        cm.upsample_refine(dec_dims[3]), 
        # 8x8
        *[cm.BaseBlockv1(dec_dims[3]) for _ in range(8)],
        VRLVBlock(dec_dims[3], z_dims[3], enc_key='enc_s8', name='z4', enc_dim=enc_dims[-4]),
    ]
    cfg['dec_blocks'] = [
        # 1x1
        *[cm.BaseBlockv1(dec_dims[0]) for _ in range(2)],
        MergeFromEM('z1', dec_dims[0]*3, dec_dims[0]),
        *[cm.BaseBlockv1(dec_dims[0]) for _ in range(2)],
        cm.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        cm.upsample_refine(dec_dims[1]), 
        # 2x2
        *[cm.BaseBlockv1(dec_dims[1]) for _ in range(2)],
        MergeFromEM('z2', dec_dims[1]*3, dec_dims[1]),
        *[cm.BaseBlockv1(dec_dims[1]) for _ in range(2)],
        cm.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        cm.upsample_refine(dec_dims[2]), 
        # 4x4
        *[cm.BaseBlockv1(dec_dims[2]) for _ in range(3)],
        MergeFromEM('z3', dec_dims[2]*3, dec_dims[2]),
        *[cm.BaseBlockv1(dec_dims[2]) for _ in range(3)],
        cm.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        cm.upsample_refine(dec_dims[3]), 
        # 8x8
        *[cm.BaseBlockv1(dec_dims[3]) for _ in range(4)],
        MergeFromEM('z4', dec_dims[3]*3, dec_dims[3]),
        # todo pixelshuffle upsample
        *[cm.BaseBlockv1(dec_dims[3]) for _ in range(4)],
        cm.WConv2d(dec_dims[3], enc_dims[0], kernel_size=3),
        nn.PixelShuffle(8),
    ]
    cfg['z_dims'] = z_dims

    model = VariableRateLossyVAE(cfg)

    return model