from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as tnf

from . import layers as cm
from . import entropy
from .bounds import UpperBound

class LatentVariableBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discrete_gaussian = entropy.DiscretizedGaussian()
        self.requires_dict_input = True
        self.scale_exponent = 2.0

    def update(self):
        self.discrete_gaussian.update()

    def forward(self, fdict, f_ctx, layer_id=0):
        feature = fdict["feature"]
        mode = fdict["mode"]
        ctx_feature = f_ctx
        lmb_em = torch.exp(fdict[f"em{layer_id + 1}_lmb"])
        ctx_feature_modulated = ctx_feature * lmb_em
        pm, pv = self.transform_prior(ctx_feature_modulated)
        z_scale = torch.exp(fdict[f"z{layer_id + 1}_scale"])
        pm_scaled = pm / z_scale
        multiplier = float(getattr(self, "entropy_scale_multiplier", 1.0))
        pv_scaled = pv / (z_scale ** self.scale_exponent) * multiplier

        if mode == "compress":
            enc_feature = fdict["all_features"][self.enc_key]
            enc_feature_modulated = enc_feature * torch.exp(fdict[f"enc{layer_id + 1}_lmb"])
            qm = self.transform_posterior(feature, enc_feature_modulated)
            qm_scaled = qm / z_scale
            self.update()
            indexes = self.discrete_gaussian.build_indexes(pv_scaled)
            fdict["bit_strings"].append(
                self.discrete_gaussian.compress(qm_scaled, indexes, means=pm_scaled))
            z_scaled = self.discrete_gaussian.quantize(
                qm_scaled, mode="dequantize", means=pm_scaled)
            z = z_scaled * z_scale
        elif mode == "decompress":
            strings = fdict["bit_strings"].pop(0)
            indexes = self.discrete_gaussian.build_indexes(pv_scaled)
            z_scaled = self.discrete_gaussian.decompress(strings, indexes, means=pm_scaled)
            z = z_scaled * z_scale
        else:
            raise ValueError(f"Unsupported runtime mode: {mode}")

        feature = feature + self.z_proj(z)
        fdict["feature"] = feature
        fdict["all_features"][f"{self.name}_z"] = z
        fdict["all_features"][f"{self.name}_out"] = feature
        fdict["all_features"][f"{self.name}_ctx"] = ctx_feature
        return fdict


class VRLVBlock(LatentVariableBlock):
    def __init__(self, dim, zdim, enc_key, enc_dim, name=None):
        super().__init__()
        self.in_channels  = dim
        self.enc_key = enc_key
        self.name = name
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
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = cm.FeatureExtractorWithEmbedding(config.pop('enc_blocks'))
        self.em_blocks = nn.ModuleList(config.pop('em_blocks'))
        self.dec_blocks = nn.ModuleList(config.pop('dec_blocks'))

        width = self.em_blocks[2].in_channels
        self.em_bias = nn.Parameter(torch.zeros(1, width, 1, 1))
        self.dec_bias = nn.Parameter(torch.zeros(1, width, 1, 1))

        self.max_stride = config['max_stride']
        z_dims = config['z_dims']
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

        self.z1_scale_embedding = nn.Parameter(torch.zeros((self.q_num, z_dims[0], 1, 1)))
        self.z2_scale_embedding = nn.Parameter(torch.zeros((self.q_num, z_dims[1], 1, 1)))
        self.z3_scale_embedding = nn.Parameter(torch.zeros((self.q_num, z_dims[2], 1, 1)))
        self.z4_scale_embedding = nn.Parameter(torch.zeros((self.q_num, z_dims[3], 1, 1)))


    def quality_indices(self, qp, batch_size=1):
        qp = int(qp)
        if not 0 <= qp < self.q_num:
            raise ValueError(f"QP must be in [0, {self.q_num - 1}]")
        return [qp] * int(batch_size)


    def get_initial_fdict(self, layer_qp, bias_bhw):
        """Create the feature dictionary used by the entropy model."""
        fdict = dict() # a feature dictionary containing all features
        nB, nH, nW = bias_bhw

        enc1_lmb, enc2_lmb, enc3_lmb, enc4_lmb = [], [], [], []
        em1_lmb, em2_lmb, em3_lmb, em4_lmb = [], [], [], []
        dec1_lmb, dec2_lmb, dec3_lmb, dec4_lmb = [], [], [], []
        z1_scale, z2_scale, z3_scale, z4_scale = [], [], [], []

        for j in range(nB):
            qp = layer_qp[j]
            qps = [int(qp)] * 4
            enc1_lmb.append(self.enc1_lmb_embedding[qps[0]])
            enc2_lmb.append(self.enc2_lmb_embedding[qps[1]])
            enc3_lmb.append(self.enc3_lmb_embedding[qps[2]])
            enc4_lmb.append(self.enc4_lmb_embedding[qps[3]])

            em1_lmb.append(self.em1_lmb_embedding[qps[0]])
            em2_lmb.append(self.em2_lmb_embedding[qps[1]])
            em3_lmb.append(self.em3_lmb_embedding[qps[2]])
            em4_lmb.append(self.em4_lmb_embedding[qps[3]])

            dec1_lmb.append(self.dec1_lmb_embedding[qps[0]])
            dec2_lmb.append(self.dec2_lmb_embedding[qps[1]])
            dec3_lmb.append(self.dec3_lmb_embedding[qps[2]])
            dec4_lmb.append(self.dec4_lmb_embedding[qps[3]])

            z1_scale.append(self.z1_scale_embedding[qps[0]])
            z2_scale.append(self.z2_scale_embedding[qps[1]])
            z3_scale.append(self.z3_scale_embedding[qps[2]])
            z4_scale.append(self.z4_scale_embedding[qps[3]])

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

        fdict['all_features'] = OrderedDict()

        fdict['feature'] = self.em_bias.expand(nB, -1, nH, nW)
        fdict['dec_feature'] = self.dec_bias.expand(nB, -1, nH, nW)
        fdict['bit_strings'] = []
        return fdict

    def forward_bottomup(self, im, layer_qp):
        bias_bhw = (im.shape[0], im.shape[2]//self.max_stride, im.shape[3]//self.max_stride)
        fdict = self.get_initial_fdict(layer_qp, bias_bhw)
        fdict['all_features'] = self.encoder(im)
        return fdict, im

    def forward_em(self, fdict, mode='compress'):
        fdict['mode'] = mode

        layer_id = 0
        for _, block in enumerate(self.em_blocks):
            if getattr(block, 'requires_dict_input', False):
                fdict = block(fdict, fdict['feature'], layer_id=layer_id)
                layer_id += 1
            else:
                fdict['feature'] = block(fdict['feature'])

        return fdict

    def forward_topdown(self, fdict, mode='compress'):
        fdict = self.forward_em(fdict, mode)

        layer_id = 0
        for _, block in enumerate(self.dec_blocks):
            if getattr(block, 'requires_dict_input', False):
                fdict = block(fdict, layer_id=layer_id)
                layer_id += 1
            else:
                fdict['dec_feature'] = block(fdict['dec_feature'])

        fdict['x_hat'] = fdict.pop('dec_feature') # rename 'feature' to 'x_hat'
        return fdict




def dhvc(internal_padding='zeros'):
    cfg = dict()

    cfg['max_stride'] = 64
    ch = 128
    enc_dims = [192, ch*3, ch*4, ch*4, ch*4]

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
        VRLVBlock(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_dim=enc_dims[-1], name='z1'),
        *[cm.BaseBlockv1(dec_dims[0]) for _ in range(2)],
        cm.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        cm.upsample_refine(dec_dims[1]),
        # 2x2
        *[cm.BaseBlockv1(dec_dims[1]) for _ in range(2)],
        VRLVBlock(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_dim=enc_dims[-2], name='z2'),
        *[cm.BaseBlockv1(dec_dims[1]) for _ in range(2)],
        cm.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        cm.upsample_refine(dec_dims[2]),
        # 4x4
        *[cm.BaseBlockv1(dec_dims[2]) for _ in range(3)],
        VRLVBlock(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_dim=enc_dims[-3], name='z3'),
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
    if internal_padding not in ('zeros', 'replicate'):
        raise ValueError('internal_padding must be zeros or replicate')
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, cm.WConv2d)):
            module.padding_mode = internal_padding
    return model
