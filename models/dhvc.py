import math
import time
import struct
import models.utils as utils
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as tnf

from .common import FeatureExtractor, MyConvNeXtBlock, TemporalLatentBlock, \
    patch_downsample, patch_upsample, myconvnext_down


class HierarchicalVideoCodec(nn.Module):
    log2_e = math.log2(math.e)

    def __init__(self, config: dict):
        super().__init__()
        # feature extractor (bottom-up path)
        self.encoder = FeatureExtractor(config.pop('enc_blocks'))
        # latent variable blocks (top-down path)
        self.dec_blocks = nn.ModuleList(config.pop('dec_blocks'))

        self.fea_blocks = nn.ModuleList(config.pop('fea_blocks'))

        width = self.dec_blocks[0].in_channels
        self.bias = nn.Parameter(torch.zeros(1, width, 1, 1))

        self.bias_list = []
        for z_dim, z_num in zip(config['z_dims'], config['z_nums']):
            for _ in range(z_num):
                bias = nn.Parameter(torch.zeros(1, z_dim, 1, 1))
                self.bias_list.append(bias)
        self.bias_list = nn.ParameterList(self.bias_list)
        self.z_nums = config['z_nums']

        self.lmb = config['lambda']
        self.compressing = False

    def get_bias(self, bhw_repeat=(1,1,1)):
        nB, nH, nW = bhw_repeat
        feature = self.bias.expand(nB, -1, nH, nW)
        return feature
    
    def get_temp_bias(self, frame):
        nB, _, nH, nW = frame.shape
        nH, nW = nH//64, nW//64
        z_list = []
        i = 0
        for z_num in self.z_nums:
            for _ in range(z_num):
                z = self.bias_list[i].expand(nB, -1, nH, nW)
                z_list.append(z)
                i += 1
            nH, nW = nH*2, nW*2
        return z_list

    def mse_loss(self, fake, real):
        assert fake.shape == real.shape
        return tnf.mse_loss(fake, real, reduction='none').mean(dim=(1,2,3))

    def forward(self, frames):        
        # initialize statistics for training and logging
        stats = OrderedDict()
        stats_keys = ['loss', 'bpp', 'psnr']
        for key in stats_keys:
            stats[key] = 0.0

        z_list = self.get_temp_bias(frames[0])
        z_lists = [z_list, z_list]

        for _, frame in enumerate(frames):
            frame_stats = self.forward_frame(frame, z_lists, get_latent=True)
            z_lists = [z_lists[-1], frame_stats['z']]

            # logging
            stats['loss'] = stats['loss'] + frame_stats['loss']
            stats['bpp']     += float(frame_stats['bpp'])
            stats['psnr']    += float(frame_stats['psnr'])

        # average over p-frames only
        for key in stats_keys:
            stats[key] = stats[key] / len(frames)

        return stats

    def forward_frame(self, x, z_lists, get_latent=False, return_rec=False):
        _, _, imH, imW = x.shape
        enc_features = self.encoder(x)
        stats_all = []
        z_new_list = []
        nB, _, nH, nW = enc_features[min(enc_features.keys())].shape
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        dec_feature = None
        i = 0
        for _, (block, fea_block) in enumerate(zip(self.dec_blocks, self.fea_blocks)):
            if getattr(block, 'is_latent_block', False):
                key = int(feature.shape[2])
                f_enc = enc_features[key]
                f_ctx = [z_list[i] for z_list in z_lists]
                feature, dec_feature, stats = block(feature, dec_feature, f_ctx, enc_feature=f_enc, mode='trainval',
                                                        get_latent=get_latent)
                if get_latent:
                    z_new_list.append(stats['z'])
                stats_all.append(stats)
                i += 1
            else:
                dec_feature = block(dec_feature)
            feature = fea_block(feature)
        x_hat = dec_feature

        # ================ Compute Loss ================
        # rate
        kl_divergences = [stat['kl'].sum(dim=(1, 2, 3)) for stat in stats_all]
        ndims = float(imH * imW)
        kl = self.log2_e * sum(kl_divergences) / ndims # nats per dimension
        # distortion
        distortion = self.mse_loss(x_hat, x)
        # rate + distortion
        loss = kl + self.lmb * distortion
        loss = loss.mean(0)

        stats = OrderedDict()
        stats['loss'] = loss

        # ================ Logging ================
        with torch.no_grad():
            # for training print
            stats['bpp'] = kl.mean(0).item()
            stats['mse'] = distortion.mean(0).item()
            x_hat = x_hat.detach()
            im_mse = tnf.mse_loss(x_hat, x, reduction='mean')
            psnr = -10 * math.log10(im_mse.item())
            stats['psnr'] = psnr

        if get_latent:
            stats['z'] = z_new_list
        if return_rec:
            stats['im_hat'] = x_hat
        return stats

    def compress_mode(self, mode=True):
        if mode:
            for block in self.dec_blocks:
                if hasattr(block, 'update'):
                    block.update()
        self.compressing = mode

    @torch.no_grad()
    def compress(self, x, z_lists, get_latent=False):
        enc_features = self.encoder(x)

        nB, _, nH, nW = enc_features[min(enc_features.keys())].shape
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        strings_all = []
        head_info = struct.pack('3H', nB, nH, nW)
        dec_feature = None
        i = 0
        for _, (block, fea_block) in enumerate(zip(self.dec_blocks, self.fea_blocks)):
            if getattr(block, 'is_latent_block', False):
                key = int(feature.shape[2])
                f_enc = enc_features[key]
                f_ctx = [z_list[i] for z_list in z_lists]
                feature, dec_feature, stats = block(feature, dec_feature, f_ctx, enc_feature=f_enc, mode='compress',
                                                    get_latent=get_latent)
                strings_all.extend(stats['strings'])
                i += 1
            else:
                dec_feature = block(dec_feature)
            feature = fea_block(feature)
        strings_all = utils.pack_byte_strings(strings_all)
        return head_info, strings_all

    @torch.no_grad()
    def decompress(self, head_info, compressed_object, z_lists, num_latents=5, get_latent=False):
        nB, nH, nW = struct.unpack('3H', head_info) # smallest feature shape
        compressed_object = utils.unpack_byte_string(compressed_object)
        compressed_object = [[s,] for s in compressed_object]
        z_new_list = []
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        dec_feature = None
        str_i = 0
        i = 0
        for _, (block, fea_block) in enumerate(zip(self.dec_blocks, self.fea_blocks)):
            if getattr(block, 'is_latent_block', False):
                strs_batch = compressed_object[str_i]
                f_ctx = [z_list[i] for z_list in z_lists]
                feature, dec_feature, stats = block(feature, dec_feature, f_ctx, mode='decompress',
                                                        strings=strs_batch, get_latent=get_latent)

                if get_latent:
                    z_new_list.append(stats['z'])
                str_i += 1
                i += 1
            else:
                dec_feature = block(dec_feature)
            feature = fea_block(feature)
        x_hat = dec_feature
        if get_latent:
            return x_hat, z_new_list
        return x_hat
    

def dhvc_base(lamb=2048):
    cfg = dict()

    ch = 128
    enc_dims = [192, ch*3, ch*4, ch*4, ch*4]
    dec_dims = [ch*4, ch*4, ch*3, ch*2, ch*1]
    z_dims = [32, 32, 96, 8]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[MyConvNeXtBlock(enc_dims[0], kernel_size=7) for _ in range(6)],
        myconvnext_down(enc_dims[0], enc_dims[1]),
        # 8x8
        *[MyConvNeXtBlock(enc_dims[1], kernel_size=7) for _ in range(2)],
        myconvnext_down(enc_dims[1], enc_dims[2]),
        # 4x4
        *[MyConvNeXtBlock(enc_dims[2], kernel_size=5) for _ in range(2)],
        myconvnext_down(enc_dims[2], enc_dims[3]),
        # 2x2
        *[MyConvNeXtBlock(enc_dims[3], kernel_size=3) for _ in range(4)],
        myconvnext_down(enc_dims[3], enc_dims[4]),
        # 1x1
        *[MyConvNeXtBlock(enc_dims[4], kernel_size=1) for _ in range(4)],
    ]

    cfg['dec_blocks'] = [
        # 1x1
        *[TemporalLatentBlock(dec_dims[0], z_dims[0], enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=4) for _ in range(1)],
        MyConvNeXtBlock(dec_dims[0], kernel_size=1, mlp_ratio=4),
        patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        MyConvNeXtBlock(dec_dims[1], kernel_size=3, mlp_ratio=3),
        *[TemporalLatentBlock(dec_dims[1], z_dims[1], enc_width=enc_dims[-2], kernel_size=3, mlp_ratio=3) for _ in range(2)],
        MyConvNeXtBlock(dec_dims[1], kernel_size=3, mlp_ratio=3),
        patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        MyConvNeXtBlock(dec_dims[2], kernel_size=5, mlp_ratio=2),
        *[TemporalLatentBlock(dec_dims[2], z_dims[2], enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(1)],
        MyConvNeXtBlock(dec_dims[2], kernel_size=5, mlp_ratio=2),
        patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        MyConvNeXtBlock(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        *[TemporalLatentBlock(dec_dims[3], z_dims[3], enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(1)],
        MyConvNeXtBlock(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[MyConvNeXtBlock(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(8)],
        patch_upsample(dec_dims[4], im_channels, rate=4)
    ]

    cfg['fea_blocks'] = [
        # 1x1
        *[nn.Identity() for _ in range(1)],
        MyConvNeXtBlock(dec_dims[0], kernel_size=1, mlp_ratio=4),
        patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        MyConvNeXtBlock(dec_dims[1], kernel_size=3, mlp_ratio=3),
        *[nn.Identity() for _ in range(2)],
        MyConvNeXtBlock(dec_dims[1], kernel_size=3, mlp_ratio=3),
        patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        MyConvNeXtBlock(dec_dims[2], kernel_size=5, mlp_ratio=2),
        *[nn.Identity() for _ in range(1)],
        MyConvNeXtBlock(dec_dims[2], kernel_size=5, mlp_ratio=2),
        patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        MyConvNeXtBlock(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        *[nn.Identity() for _ in range(1)],
        MyConvNeXtBlock(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[MyConvNeXtBlock(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(8)],
        patch_upsample(dec_dims[4], im_channels, rate=4)
    ]

    cfg['z_dims'] = dec_dims
    cfg['z_nums'] = [1, 2, 1, 1]
    cfg['lambda'] = lamb

    model = HierarchicalVideoCodec(cfg)
    return model


if __name__ == "__main__":
    model = dhvc_base()
    im = torch.ones([1, 3, 64, 64])
    frames = []
    frames.append(im)
    frames.append(im)
    frames.append(im)
    out = model(frames)



