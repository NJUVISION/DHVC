import argparse
import sys
import logging
import os
import time
import json
import pickle
import struct
from pathlib import Path
from collections import OrderedDict
from ptflops import get_model_complexity_info

import cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from models.dhvc import dhvc_base as DHVC

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def evaluate_one_video(args, quality, frame_dir):

    net = DHVC()

    macs, params = get_model_complexity_info(
        net, (1, 3, 256, 256), as_strings=False, print_per_layer_stat=False
    )
    logging.info("MACs/pixel:" + str(macs / (256**2)))
    logging.info("params:" + str(params))

    snapshot = torch.load(args.checkpoint_path)
    net.load_state_dict(snapshot['state_dict'])
    
    net.to(device).eval()
    net.compress_mode()

    # sanity check
    results_dir = Path(args.results_dir)
    assert results_dir.is_dir(), f'results_dir={results_dir} does not exist'
    save_path = results_dir / f'q{quality}.json'
    logging.info(f'starting q={quality}, frame_dir={frame_dir}, save_path={save_path}')

    tic = time.time()

    frame_dir = Path(frame_dir)
    _str = f'{args.test_dataset}-q{quality}-gop{args.gop}-num{args.num_frames}'
    save_bit_path = Path(f'cache/{_str}/{frame_dir.stem}.bits')
    if not save_bit_path.parent.is_dir():
        save_bit_path.parent.mkdir(parents=True, exist_ok=True)

    # # compute metrics
    ori_frame_paths = list(Path(frame_dir).glob('*.png'))
    ori_frame_paths.sort()
    if args.num_frames == None:
        num_frames = len(ori_frame_paths)
    else:
        num_frames = args.num_frames
    img_height, img_width = cv2.imread(str(ori_frame_paths[0])).shape[:2]

    # compute psnr
    sum_bpp = 0.0
    sum_psnr = 0.0
    for fi, ori_fp in enumerate(ori_frame_paths[:num_frames]):
        # read an original frame
        x = cv2.cvtColor(cv2.imread(str(ori_fp)), cv2.COLOR_BGR2RGB) / 255.
        assert x.shape == (img_height, img_width, 3)

        x = torch.FloatTensor(x).permute(2, 0, 1).unsqueeze(0).to(device)

        p = 64
        x_pad = pad(x, p)

        if fi % args.gop == 0:
            z_list = net.get_temp_bias(x_pad)
            z_lists = [z_list, z_list]
        with torch.no_grad():
            head_info, compressed_strings = net.compress(x_pad, z_lists, get_latent=True)
            with open(save_bit_path, "wb") as f:
                f.write(head_info + compressed_strings)
            with open(save_bit_path, 'rb') as f:
                head_info = f.read(6)
                compressed_strings = f.read()
            rec_pad, z_new_list = net.decompress(head_info, compressed_strings, z_lists, get_latent=True)
            z_lists = [z_lists[-1], z_new_list]
            rec_pad = rec_pad.clamp(0, 1)

        rec = crop(rec_pad, (img_height, img_width))
        mse = torch.mean((x - rec)**2).item()
        psnr = -10 * np.log10(mse)
        bpp = float(filesize(save_bit_path)) * 8 / (img_height * img_width)
        sum_bpp += bpp
        sum_psnr += psnr
        print('frame:', fi, 'bpp:', bpp, 'psnr:', psnr)
    
    # compute bpp
    avg_bpp = sum_bpp / num_frames
    avg_psnr = sum_psnr / num_frames

    stats = OrderedDict()
    stats['video']   = str(frame_dir)
    stats['quality'] = quality
    stats['bpp']     = avg_bpp
    stats['psnr']    = avg_psnr

    # save results
    if save_path.is_file():
        with open(save_path, mode='r') as f:
            all_seq_results = json.load(fp=f)
        assert isinstance(all_seq_results, list)
        all_seq_results.append(stats)
    else:
        all_seq_results = [stats]
    with open(save_path, mode='w') as f:
        json.dump(all_seq_results, fp=f, indent=2)

    elapsed = time.time() - tic
    msg = f'q={quality}, time={elapsed:.1f}s. '
    msg += f'evaluate {num_frames} out of {len(ori_frame_paths)} frames. '
    msg += f'frame_dir.stem={frame_dir.stem}, bpp={avg_bpp}, psnr={avg_psnr}'
    logging.info('\u001b[92m' + msg + '\u001b[0m')
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--test-dataset', type=str, default='uvg')
    parser.add_argument('-c', '--checkpoint-path', type=str, default='./pretrained/3/checkpoint_best_loss.pth.tar')
    parser.add_argument('-p', '--test-dataset-path', type=str, default='./datasets/UVG/PNG')
    parser.add_argument('-q', '--quality',      type=int, default=[6], nargs='+')
    parser.add_argument('-g', '--gop',          type=int, default=32)
    parser.add_argument('-f', '--num-frames',   type=int, default=96)
    parser.add_argument("--intra", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    suffix = 'allframes' if (args.num_frames is None) else f'first{args.num_frames}'
    results_dir = Path(f'runs/results/{args.test_dataset}-gop{args.gop}-{suffix}')
    if not results_dir.is_dir():
        results_dir.mkdir(parents=True)

    # init logging
    setup_logger(str(results_dir) + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')

    logging.info(f'Saving results to {results_dir}')
    args.results_dir = results_dir

    video_frame_dirs = list(Path(args.test_dataset_path).glob('*/'))
    video_frame_dirs.sort()
    logging.info(f'Total {len(video_frame_dirs)} sequences')

    mp_results = []
    # enumerate all quality
    for q in args.quality:
        sum_psnr = 0.0
        sum_bpp = 0.0
        for i, vfd in enumerate(video_frame_dirs):
            mp_results = evaluate_one_video(args, q, vfd)
            sum_psnr += mp_results['psnr']
            sum_bpp += mp_results['bpp']
        
        avg_bpp = sum_bpp / len(video_frame_dirs)
        avg_psnr = sum_psnr / len(video_frame_dirs)
        logging.info(f'AVERAGE BPP: {avg_bpp}, AVERAGE PSNR: {avg_psnr}')


if __name__ == "__main__":
    main()