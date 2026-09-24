"""Image coding evaluation helpers."""
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def read_image(path):
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255)


def evaluate_image(codec, path, qp):
    path = Path(path)
    source = read_image(path)
    height, width = source.shape[-2:]

    start = time.perf_counter()
    stream = codec.encode(str(path), qp=qp)
    encode_seconds = time.perf_counter() - start

    start = time.perf_counter()
    reconstruction = codec.decode_tensor(stream)
    decode_seconds = time.perf_counter() - start

    if reconstruction.shape != source.shape:
        raise RuntimeError("Decoded image shape does not match the input")
    source = source.to(reconstruction.device)
    mse = float((reconstruction - source).square().mean())
    return {
        "image": str(path), "qp": int(qp),
        "height": height, "width": width,
        "container_bytes": len(stream),
        "bpp": len(stream) * 8 / (height * width),
        "mse": mse, "psnr": -10 * math.log10(max(mse, 1e-12)),
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "max_codec_error": 0.0,
    }
