# models/entropy_coding_fast.py
import torch
import torch.nn as nn
import math

try:
    from models.entropy_models import GaussianEncoder, EntropyCoder
    from models.common_model import CompressionModel 
    from models.cuda_inference import (
        process_with_mask, 
        add_and_multiply, 
        combine_for_reading_2x, 
        restore_y_2x
    )
except ImportError:
    pass


class DCVCEntropyEngine(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.gaussian_encoder = GaussianEncoder()
        self.entropy_coder = None 
        self.register_buffer('sqrt_2', torch.tensor(math.sqrt(2.0)), persistent=False)

    def update(self):
        self.entropy_coder = EntropyCoder()
        self.gaussian_encoder.update(self.entropy_coder)

    def get_stream(self):
        return self.entropy_coder.get_encoded_stream()

    def set_stream(self, stream):
        self.entropy_coder.set_stream(stream)

    def forward_training(self, y, q_step, scales, means):
        noise = torch.empty_like(y).uniform_(-0.5, 0.5)
        y_noisy = y + noise * q_step

        scales = torch.clamp(scales, min=1e-6)

        centered = y_noisy - means
        half_q = 0.5 * q_step

        upper = (centered + half_q) / scales
        lower = (centered - half_q) / scales
        cdf_upper = torch.erf(upper / self.sqrt_2)
        cdf_lower = torch.erf(lower / self.sqrt_2)
        probs = 0.5 * (cdf_upper - cdf_lower)

        probs = torch.clamp(probs, min=1e-9)
        
        log_prob = torch.log(probs)
        
        return y_noisy, log_prob

# ===========================================================================
# 2. Context Models (Checkerboard & True Channel Context)
# ===========================================================================
class AttentionContextBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = x * self.ca(x)
        return x + res


class HeterogeneousContext(nn.Module):
    def __init__(self, dim, mode='checkerboard'):
        super().__init__()
        self.mode = mode
        self.helper = CompressionModel(dim)
        
        if mode == 'checkerboard':
            self.context_net = nn.Sequential(
                nn.Conv2d(dim, dim * 2, 3, 1, 1),
                AttentionContextBlock(dim * 2),
                AttentionContextBlock(dim * 2),
                AttentionContextBlock(dim * 2),
                nn.Conv2d(dim * 2, dim * 2, 3, 1, 1)
            )
        elif mode == 'channel':
            # 真正的通道上下文网络
            # 输入：第一块解码的隐变量 (half_dim)
            # 输出：第二块的 delta_scale 和 delta_mean (out_dim)
            half_dim = dim // 2
            out_dim = (dim - half_dim) * 2 # 严谨处理可能的奇数维度情况
            
            self.context_net = nn.Sequential(
                nn.Conv2d(half_dim, half_dim, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(half_dim, half_dim, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(half_dim, out_dim, 3, 1, 1)
            )

    def get_masks(self, shape, device):
        return self.helper.get_mask_2x(shape[0], shape[1], shape[2], shape[3], torch.float32, device)

    def compress(self, y, q_step, scales, means, entropy_engine):
        B, C, H, W = y.shape
        
        if self.mode == 'channel':
            half_c = C // 2
            # 1. 均分通道为两块 (严格对齐维度)
            y_0, y_1 = torch.split(y, [half_c, C - half_c], dim=1)
            s_0, s_1 = torch.split(scales, [half_c, C - half_c], dim=1)
            m_0, m_1 = torch.split(means, [half_c, C - half_c], dim=1)

            # 2. 生成全 1 Mask 兼容底层 C++ 算子 (shape: B, 1, H, W)
            mask_all = torch.ones((B, 1, H, W), device=y.device)

            # 3. 编码第一块 (无额外上下文)
            res_0, y_q_0, y_hat_0, s_hat_0 = process_with_mask(y_0, s_0, m_0, mask_all)
            entropy_engine.gaussian_encoder.encode_y(y_q_0, s_hat_0)

            # 4. 利用第一块的重建特征生成第二块的先验修正值
            ctx_params = self.context_net(y_hat_0)
            delta_scales_1, delta_means_1 = ctx_params.chunk(2, 1)
            curr_scales_1 = s_1 + delta_scales_1
            curr_means_1 = m_1 + delta_means_1

            # 5. 编码第二块 (享受了第一块的通道级先验)
            res_1, y_q_1, y_hat_1, s_hat_1 = process_with_mask(y_1, curr_scales_1, curr_means_1, mask_all)
            entropy_engine.gaussian_encoder.encode_y(y_q_1, s_hat_1)

            # 6. 在通道维度拼合
            y_hat = torch.cat([y_hat_0, y_hat_1], dim=1)
            return y_hat * q_step

        else:
            # Checkerboard 逻辑保持原有设计
            mask_0, mask_1 = self.get_masks((B, C, H, W), y.device)

            res_0, y_q_0, y_hat_0, s_hat_0 = process_with_mask(y, scales, means, mask_0)
            entropy_engine.gaussian_encoder.encode_y(y_q_0, s_hat_0)

            ctx_params = self.context_net(y_hat_0)
            delta_scales, delta_means = ctx_params.chunk(2, 1)
            curr_scales = scales + delta_scales
            curr_means = means + delta_means

            res_1, y_q_1, y_hat_1, s_hat_1 = process_with_mask(y, curr_scales, curr_means, mask_1)
            entropy_engine.gaussian_encoder.encode_y(y_q_1, s_hat_1)

            y_hat = add_and_multiply(y_hat_0, y_hat_1, q_step)
            return y_hat

    def decompress(self, scales, means, q_step, entropy_engine):
        B, C, H, W = scales.shape
        
        if self.mode == 'channel':
            half_c = C // 2
            s_0, s_1 = torch.split(scales, [half_c, C - half_c], dim=1)
            m_0, m_1 = torch.split(means, [half_c, C - half_c], dim=1)

            mask_all = torch.ones((B, 1, H, W), device=scales.device)

            # 1. 解码第一块
            scales_r_0 = combine_for_reading_2x(s_0, mask_all)
            y_q_r_0 = entropy_engine.gaussian_encoder.decode_and_get_y(scales_r_0, m_0.dtype, m_0.device)
            y_hat_0 = restore_y_2x(y_q_r_0, m_0, mask_all)

            # 2. 生成第二块上下文
            ctx_params = self.context_net(y_hat_0)
            delta_scales_1, delta_means_1 = ctx_params.chunk(2, 1)
            curr_scales_1 = s_1 + delta_scales_1
            curr_means_1 = m_1 + delta_means_1

            # 3. 解码第二块
            scales_r_1 = combine_for_reading_2x(curr_scales_1, mask_all)
            y_q_r_1 = entropy_engine.gaussian_encoder.decode_and_get_y(scales_r_1, curr_means_1.dtype, curr_means_1.device)
            y_hat_1 = restore_y_2x(y_q_r_1, curr_means_1, mask_all)

            y_hat = torch.cat([y_hat_0, y_hat_1], dim=1)
            return y_hat * q_step
            
        else: 
            mask_0, mask_1 = self.get_masks((B, C, H, W), scales.device)

            scales_r_0 = combine_for_reading_2x(scales, mask_0)
            y_q_r_0 = entropy_engine.gaussian_encoder.decode_and_get_y(scales_r_0, means.dtype, means.device)
            y_hat_0 = restore_y_2x(y_q_r_0, means, mask_0)

            ctx_params = self.context_net(y_hat_0)
            delta_scales, delta_means = ctx_params.chunk(2, 1)
            curr_scales = scales + delta_scales
            curr_means = means + delta_means

            scales_r_1 = combine_for_reading_2x(curr_scales, mask_1)
            y_q_r_1 = entropy_engine.gaussian_encoder.decode_and_get_y(scales_r_1, means.dtype, means.device)
            y_hat_1 = restore_y_2x(y_q_r_1, curr_means, mask_1)

            y_hat = add_and_multiply(y_hat_0, y_hat_1, q_step)
            return y_hat

    def forward_training(self, y, q_step, scales, means, entropy_engine):
        B, C, H, W = y.shape
        
        if self.mode == 'channel':
            half_c = C // 2
            # 切分所有的条件参数
            y_0, y_1 = torch.split(y, [half_c, C - half_c], dim=1)
            s_0, s_1 = torch.split(scales, [half_c, C - half_c], dim=1)
            m_0, m_1 = torch.split(means, [half_c, C - half_c], dim=1)
            
            # 【修复点】：增加对 q_step 的同步切分！
            q_0, q_1 = torch.split(q_step, [half_c, C - half_c], dim=1)

            # 1. 训练第一块，传入匹配的 q_0
            y_noisy_0, log_prob_0 = entropy_engine.forward_training(y_0, q_0, s_0, m_0)
            
            # 2. 用第一块带有均匀噪声的特征生成上下文
            ctx_params = self.context_net(y_noisy_0)
            delta_scales_1, delta_means_1 = ctx_params.chunk(2, 1)
            curr_scales_1 = s_1 + delta_scales_1
            curr_means_1 = m_1 + delta_means_1

            # 3. 训练第二块，传入匹配的 q_1
            y_noisy_1, log_prob_1 = entropy_engine.forward_training(y_1, q_1, curr_scales_1, curr_means_1)

            # 4. 在通道维度合并输出
            y_hat = torch.cat([y_noisy_0, y_noisy_1], dim=1)
            log_prob = torch.cat([log_prob_0, log_prob_1], dim=1)
            
            return y_hat, log_prob
            
        else:
            # 原有的 Checkerboard 训练逻辑保持不变
            mask_0, mask_1 = self.get_masks((B, C, H, W), y.device)
            y_noisy_0, log_prob_0 = entropy_engine.forward_training(y, q_step, scales, means)
            y_hat_0 = y_noisy_0 * mask_0 

            ctx_params = self.context_net(y_hat_0)
            delta_scales, delta_means = ctx_params.chunk(2, 1)
            curr_scales = scales + delta_scales
            curr_means = means + delta_means

            y_noisy_1, log_prob_1 = entropy_engine.forward_training(y, q_step, curr_scales, curr_means)

            y_hat = y_hat_0 + (y_noisy_1 * mask_1)
            log_prob = (log_prob_0 * mask_0) + (log_prob_1 * mask_1)

            return y_hat, log_prob