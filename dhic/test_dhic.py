import argparse
import sys
import logging
import os
import time
import json
import pickle
import random
from pathlib import Path
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import gridspec

import cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from pytorch_msssim import ms_ssim

from models.dhvc_dec_intra_var_rate import dhvc as DHVC

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
plt.rcParams['font.family'] = 'serif'
z_dims = [320, 256, 128, 64]


def safe_load_pretrained(checkpoint_path, net, map_location='cpu'):
    snapshot = torch.load(checkpoint_path, map_location=map_location)
    pretrained_dict = snapshot.get('state_dict', snapshot)

    model_dict = net.state_dict()

    unexpected_keys = [k for k in pretrained_dict.keys() if k not in model_dict]
    missing_keys   = [k for k in model_dict.keys()   if k not in pretrained_dict]

    common_keys = set(pretrained_dict.keys()) & set(model_dict.keys())
    size_mismatch = [
        (k, pretrained_dict[k].shape, model_dict[k].shape)
        for k in common_keys
        if pretrained_dict[k].shape != model_dict[k].shape
    ]

    filtered_dict = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }

    load_result = net.load_state_dict(filtered_dict, strict=False)
    loaded_keys = [k for k in filtered_dict.keys() if k not in load_result.unexpected_keys]

    print(f"Unexpected keys (in pretrained but not in model): {unexpected_keys}")
    print(f"Missing   keys (in model but not in pretrained): {missing_keys}")
    if size_mismatch:
        print("Size mismatched keys:")
        for k, pre_shape, model_shape in size_mismatch:
            print(f"  {k:40s} | pretrained: {pre_shape} vs model: {model_shape}")

    return {
        'unexpected_keys': unexpected_keys,
        'missing_keys': missing_keys,
        'size_mismatch': size_mismatch,
        'loaded_keys': loaded_keys
    }


def fuse_all_repvggdw(model, verbose=True):
    fused = 0
    for name, m in model.named_modules():
        if hasattr(m, 'fuse') and callable(getattr(m, 'fuse')):
            try:
                m.fuse()
                fused += 1
                if verbose:
                    print(f"[fuse] fused module at: {name}")
            except Exception as e:
                print(f"[fuse WARN] failed to fuse module {name}: {e}")
    if verbose:
        print(f"[fuse] total fused: {fused}")
    return fused


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)


def pad(x, p=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )

# 【修改点1】将 net 作为参数传入，删除内部加载模型的逻辑
def evaluate_one_video(args, quality, item_path, net, decoded_frame_dir='/workspace/DHIC_LS/save/'):

    item_path = Path(item_path)
    
    # 自动判断输入是图片文件还是文件夹
    if item_path.is_file() and item_path.suffix.lower() == '.png':
        ori_frame_paths = [item_path]
        item_name = item_path.stem
    elif item_path.is_dir():
        ori_frame_paths = sorted(list(item_path.glob('*.png')))
        item_name = item_path.name
    else:
        logging.error(f'Invalid path: {item_path}')
        return None

    if len(ori_frame_paths) == 0:
        logging.warning(f'No png files found in {item_path}')
        return None

    save_dir = os.path.join(decoded_frame_dir, item_name)
    os.makedirs(save_dir, exist_ok=True)

    # sanity check
    results_dir = Path(args.results_dir)
    assert results_dir.is_dir(), f'results_dir={results_dir} does not exist'
    save_path = results_dir / f'q{quality}.json'
    logging.info(f'starting q={quality}, item={item_path}, save_path={save_path}')

    tic = time.time()

    _str = f'{args.test_dataset}-q{quality}-gop{args.gop}-num{args.num_frames}'
    save_bit_path = Path(f'cache/{_str}/{item_name}.bits')
    if not save_bit_path.parent.is_dir():
        save_bit_path.parent.mkdir(parents=True, exist_ok=True)

    f = save_bit_path.open("wb")

    if args.num_frames is None:
        num_frames = len(ori_frame_paths)
    else:
        num_frames = min(args.num_frames, len(ori_frame_paths))

    img_height, img_width = cv2.imread(str(ori_frame_paths[0])).shape[:2]

    # compute metrics
    sum_bpp = sum_bpp1 = sum_bpp2 = sum_bpp3 = sum_bpp4 = 0.0
    sum_psnr = 0.0
    sum_msssim = 0.0
    sum_mse = 0.0
    
    for fi, ori_fp in enumerate(ori_frame_paths[:num_frames]):
        # read an original frame
        x = cv2.cvtColor(cv2.imread(str(ori_fp)), cv2.COLOR_BGR2RGB) / 255.
        img_height, img_width = x.shape[:2]
        
        x = torch.FloatTensor(x).permute(2, 0, 1).unsqueeze(0).to(device)

        p = 64
        x_pad = pad(x, p)
        
        with torch.no_grad():
            frame_metrics, fdict = net(x_pad, cur_qp=3, return_fdict=True)
            sum_bpp += frame_metrics['bpp']
            sum_bpp1 += frame_metrics['bpp1']
            sum_bpp2 += frame_metrics['bpp2']
            sum_bpp3 += frame_metrics['bpp3']
            sum_bpp4 += frame_metrics['bpp4']
            rec_pad = fdict['x_hat'].clamp(0, 1)
            print('Intra frame:', fi, "Progressive Layer 0 ", 'bpp:', frame_metrics['bpp'], 'psnr:', frame_metrics['psnr'])

        rec = crop(rec_pad, (img_height, img_width))

        mse = torch.mean((x - rec)**2).item()
        psnr = -10 * np.log10(mse) if mse > 0 else 100
        msssim = ms_ssim(x, rec, data_range=1.0).item()

        sum_psnr += psnr
        sum_msssim += msssim
        sum_mse += mse
    
    f.close()
    
    # compute average metrics
    avg_bpp = sum_bpp / num_frames
    avg_bpp1 = sum_bpp1 / num_frames
    avg_bpp2 = sum_bpp2 / num_frames
    avg_bpp3 = sum_bpp3 / num_frames
    avg_bpp4 = sum_bpp4 / num_frames
    avg_psnr = sum_psnr / num_frames
    avg_msssim = sum_msssim / num_frames
    avg_mse = sum_mse / num_frames

    stats = OrderedDict()
    stats['video']   = str(item_path)
    stats['quality'] = quality
    stats['bpp']     = avg_bpp
    stats['bpp1']    = avg_bpp1
    stats['bpp2']    = avg_bpp2
    stats['bpp3']    = avg_bpp3
    stats['bpp4']    = avg_bpp4
    stats['psnr']    = avg_psnr
    stats['msssim']  = avg_msssim
    stats['mse']     = avg_mse

    # save results
    if save_path.is_file():
        with open(save_path, mode='r') as json_f:
            all_seq_results = json.load(fp=json_f)
        assert isinstance(all_seq_results, list)
        all_seq_results.append(stats)
    else:
        all_seq_results = [stats]
    with open(save_path, mode='w') as json_f:
        json.dump(all_seq_results, fp=json_f, indent=2)

    elapsed = time.time() - tic
    msg = f'q={quality}, time={elapsed:.1f}s. '
    msg += f'evaluate {num_frames} out of {len(ori_frame_paths)} frames. '
    msg += f'item={item_name}, bpp={avg_bpp:.4f}, msssim={avg_msssim:.4f}, mse={avg_mse:.6f}, psnr={avg_psnr:.4f}'
    logging.info('\u001b[92m' + msg + '\u001b[0m')
    
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--test-dataset', type=str, default='kodak')
    parser.add_argument('-q', '--quality',      type=int, default=[6], nargs='+')
    parser.add_argument('-g', '--gop',          type=int, default=-1)
    parser.add_argument('-f', '--num-frames',   type=int, default=None)
    parser.add_argument("--intra", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    frames_root = {
        'kodak': '/datasets/Kodak/',
    }

    suffix = 'allframes' if (args.num_frames is None) else f'first{args.num_frames}'
    results_dir = Path(f'runs/results/{args.test_dataset}-gop{args.gop}-{suffix}')
    if not results_dir.is_dir():
        results_dir.mkdir(parents=True)

    # init logging
    setup_logger(str(results_dir) + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    logging.info(f'Saving results to {results_dir}')
    args.results_dir = results_dir

    # 【修改点2】在主函数只加载一次模型和权重
    logging.info("Initializing model and loading checkpoint...")
    net = DHVC()
    ckpt_path = './checkpoint.pth.tar'
    snapshot = safe_load_pretrained(ckpt_path, net)
    net.load_state_dict(snapshot, strict=False)
    net.to(device).eval()
    logging.info("Model loaded successfully.")

    dataset_path = Path(frames_root[args.test_dataset])
    if not dataset_path.exists():
        logging.error(f"Dataset path does not exist: {dataset_path}")
        return

    # 动态解析测试条目：视频序列(文件夹) 或 图像序列(直接PNG文件)
    subdirs = sorted([p for p in dataset_path.iterdir() if p.is_dir()])
    if len(subdirs) > 0:
        # 数据集包含子文件夹，认定为视频数据集 (例如UVG)
        test_items = subdirs
    else:
        # 数据集直接包含PNG图片，认定为静态图像数据集 (例如Kodak)
        test_items = sorted(list(dataset_path.glob('*.png')))
        if len(test_items) == 0:
            test_items = [dataset_path] # 如果没有任何发现，退回测试目录本身以抛出适当错误
    
    logging.info(f'Total {len(test_items)} sequences/images found in {dataset_path}')
    
    # enumerate all quality
    for q in args.quality:
        sum_mse = 0.0
        sum_msssim = 0.0
        sum_bpp = sum_bpp1 = sum_bpp2 = sum_bpp3 = sum_bpp4 = 0.0
        sum_psnr = 0.0
        valid_items = 0
        
        for i, item in enumerate(test_items):
            # 【修改点3】将预加载好的 net 作为参数传入
            mp_results = evaluate_one_video(args, q, item, net)
            if mp_results is None:
                continue
            
            sum_msssim += mp_results['msssim']
            sum_mse += mp_results['mse']
            sum_psnr += mp_results['psnr']
            sum_bpp += mp_results['bpp']
            sum_bpp1 += mp_results['bpp1']
            sum_bpp2 += mp_results['bpp2']
            sum_bpp3 += mp_results['bpp3']
            sum_bpp4 += mp_results['bpp4']
            valid_items += 1
        
        if valid_items > 0:
            avg_bpp = sum_bpp / valid_items
            avg_bpp1 = sum_bpp1 / valid_items
            avg_bpp2 = sum_bpp2 / valid_items
            avg_bpp3 = sum_bpp3 / valid_items
            avg_bpp4 = sum_bpp4 / valid_items
            avg_msssim = sum_msssim / valid_items
            avg_mse = sum_mse / valid_items
            avg_psnr = sum_psnr / valid_items
            
            logging.info(f'===== Q{q} FINAL RESULTS =====')
            logging.info(f'AVERAGE BPP: {avg_bpp:.4f}, AVERAGE MSSSIM: {avg_msssim:.4f}, AVERAGE MSE: {avg_mse:.6f}, AVERAGE PSNR: {avg_psnr:.4f}')
            logging.info(f'AVERAGE BPP1: {avg_bpp1:.4f}, AVERAGE BPP2: {avg_bpp2:.4f}, AVERAGE BPP3: {avg_bpp3:.4f}, AVERAGE BPP4: {avg_bpp4:.4f}')


if __name__ == "__main__":
    main()