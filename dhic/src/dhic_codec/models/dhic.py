import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as tnf

from collections import OrderedDict

from . import common as cm
from . import entropy_coding
from .bound_ops import UpperBound


class ConditioningEmbeddings(nn.Module):
    """Variable-rate conditioning tables grouped by network stage and scale."""

    LAMBDA_STAGES = ("enc", "em", "dec")

    def __init__(self, q_num: int, lambda_dims: list[int], z_dims: list[int]):
        super().__init__()
        self.lambda_dims = lambda_dims
        self.z_dims = z_dims
        self.lambda_embeddings = nn.ParameterDict(
            {
                f"{stage}{scale}": nn.Parameter(torch.zeros(q_num, dim, 1, 1))
                for stage in self.LAMBDA_STAGES
                for scale, dim in enumerate(lambda_dims, start=1)
            }
        )
        self.z_scale_embeddings = nn.ParameterDict(
            {
                f"z{scale}": nn.Parameter(torch.zeros(q_num, dim, 1, 1))
                for scale, dim in enumerate(z_dims, start=1)
            }
        )

    def select(self, layer_qp):
        device = next(self.parameters()).device
        qp = torch.as_tensor(layer_qp, dtype=torch.long, device=device)

        selected = {}
        for stage in self.LAMBDA_STAGES:
            for scale in range(1, len(self.lambda_dims) + 1):
                key = f"{stage}{scale}"
                selected[f"{key}_lmb"] = self.lambda_embeddings[key].index_select(0, qp)
        for scale in range(1, len(self.z_dims) + 1):
            key = f"z{scale}"
            selected[f"{key}_scale"] = self.z_scale_embeddings[key].index_select(0, qp)
        return selected

    @classmethod
    def legacy_key_map(cls, num_scales: int):
        mapping = {}
        for stage in cls.LAMBDA_STAGES:
            for scale in range(1, num_scales + 1):
                mapping[f"{stage}{scale}_lmb_embedding"] = f"conditioning.lambda_embeddings.{stage}{scale}"
        for scale in range(1, num_scales + 1):
            mapping[f"z{scale}_scale_embedding"] = f"conditioning.z_scale_embeddings.z{scale}"
        return mapping


def sample_qp_and_lambda(cur_qp, batch_size, q_num=64, min_lambda=256, max_lambda=8192, train_fixed=False):
    if cur_qp is None:
        cur_qp = [q_num - 1] * batch_size if train_fixed else random.choices(range(q_num), k=batch_size)
    elif isinstance(cur_qp, int):
        cur_qp = [cur_qp] * batch_size

    q_index = torch.tensor(cur_qp, dtype=torch.float32)
    log_min = math.log(min_lambda)
    log_max = math.log(max_lambda)
    log_lambda = log_min + (q_index / (q_num - 1)) * (log_max - log_min)
    return cur_qp, torch.exp(log_lambda)


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

    def __init__(self, dim, zdim, enc_key, enc_dim, name=None, emb_dim=None, kernel_size=7):
        super().__init__()
        self.in_channels  = dim
        self.out_channels = dim
        self.enc_key = enc_key
        self.name = name

        emb_dim = emb_dim or self.default_embedding_dim
        self.enc_resblock = cm.BaseBlockv1(enc_dim)
        self.posterior = cm.BaseBlockv1(dim + enc_dim, dim, zdim)
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
        self.conditioning = ConditioningEmbeddings(
            q_num=self.q_num,
            lambda_dims=config['lambda_dims'],
            z_dims=self.z_dims,
        )
        self.register_load_state_dict_pre_hook(self._remap_legacy_conditioning_keys)

    def remap_legacy_state_dict(self, state_dict, prefix=""):
        for old_name, new_name in ConditioningEmbeddings.legacy_key_map(len(self.z_dims)).items():
            old_key = prefix + old_name
            new_key = prefix + new_name
            if old_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
        return state_dict

    def _remap_legacy_conditioning_keys(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.remap_legacy_state_dict(state_dict, prefix=prefix)


    def sample_lmb(self, cur_qp=None, min_lamda=256, max_lamda=8192, q_num=64, bs=1, train_fixed=False):
        return sample_qp_and_lambda(
            cur_qp=cur_qp,
            batch_size=bs,
            q_num=q_num,
            min_lambda=min_lamda,
            max_lambda=max_lamda,
            train_fixed=train_fixed,
        )


    def get_initial_fdict(self, layer_qp, bias_bhw, use_dispersive=False):
        fdict = dict()
        nB, nH, nW = bias_bhw
        fdict.update(self.conditioning.select(layer_qp))

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

        return fdict
    

    def forward(self, im, cur_qp=None, return_fdict=False, train_fixed=False, use_dispersive=False, kl_rate=1.0):
        B, imC, imH, imW = im.shape
        layer_qp, lmb = self.sample_lmb(cur_qp=cur_qp, train_fixed=train_fixed, bs=B)
        lmb = lmb.to(self._dummy.device, non_blocking=True)
        im = im.to(self._dummy.device)
        fdict, x = self.forward_bottomup(im, layer_qp, use_dispersive=use_dispersive)
        fdict = self.forward_topdown(fdict, mode='trainval', use_dispersive=use_dispersive)

        x_hat, kl_divs = fdict['x_hat'], fdict['kl_divs']
        kl_divs = [kl.sum(dim=(1, 2, 3)) for kl in kl_divs]
        bpp = sum(kl_divs) * self.log2_e / float(imH * imW)

        mse = tnf.mse_loss(x_hat, x, reduction='none').mean(dim=(1,2,3))
        loss = kl_rate * (bpp) + lmb * mse

        metrics = OrderedDict()
        metrics['loss'] = loss.mean(0)

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


def dhic():
    cfg = dict()
    cfg['max_stride'] = 64
    cfg['lmb_embed_dim'] = (256, 256)
    ch = 128
    enc_dims = [192, ch*3, ch*4, ch*4, ch*4]
    cfg['lambda_dims'] = [ch*4, ch*4, ch*4, ch*3]

    cfg['enc_blocks'] = [
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
        *[cm.BaseBlockv1(dec_dims[3]) for _ in range(4)],
        cm.WConv2d(dec_dims[3], enc_dims[0], kernel_size=3),
        nn.PixelShuffle(8),
    ]
    cfg['z_dims'] = z_dims

    model = VariableRateLossyVAE(cfg)

    return model
