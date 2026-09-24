# Taming Hierarchical Image Coding Optimization: A Spectral Regularization Perspective (ICLR 2026)

[![PyTorch](https://img.shields.io/badge/PyTorch-required-EE4C2C.svg)](https://pytorch.org/)
[![Conference](https://img.shields.io/badge/ICLR-2026-blue)](https://iclr.cc/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](../LICENSE.txt)

**[Taming Hierarchical Image Coding Optimization: A Spectral Regularization Perspective](https://openreview.net/forum?id=lO6I66lweK)**  
Wuyang Cong, Junqi Shi, Ming Lu, Xu Zhang, Zhan Ma  
*Nanjing University*

Official PyTorch implementation of the ICLR 2026 paper. We introduce a novel **Spectral Regularization** framework for Deep Hierarchical Image Compression (DHIC), specifically designed to overcome optimization bottlenecks in hierarchical architectures and achieved better performance.

## 💡 Abstract & Core Contributions

Hierarchical coding offers distinct advantages for learned image compression by capturing multi-scale representations, but its practical performance has been limited by cross-scale energy dispersion and spectral aliasing. 

To address this, we propose two explicit spectral regularization schemes:
1. **Intra-scale Frequency Regularization**: Encourages a smooth low-to-high frequency buildup as scales increase.
2. **Inter-scale Similarity Regularization**: Suppresses spectral aliasing across scales.

**Highlights:**

- 🚀 **Accelerated Convergence**: Speeds up the training of vanilla hierarchical models by **2.3x**.
- 📈 **State-of-the-Art Performance**: Delivers an average **20.65% rate-distortion gain** over VTM-22.0.
- ⚡ **Zero Inference Overhead**: Both regularizers are applied *only during training* and impose absolute zero additional FLOPs or parameters during testing.

---

## Installation

Requires Linux x86_64 (glibc 2.32+), Python 3.12, and a compatible PyTorch
installation.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Download the [pretrained model](https://box.nju.edu.cn/seafhttp/f/c0c24ba73569426194e1/?op=view)
and place it at `weights/model.pth`.

## Encode and decode

```python
from dhic import Codec

codec = Codec("weights/model.pth", device="cuda:0")
stream = codec.encode("input.png", qp=31)
codec.decode(stream).save("reconstruction.png")
```

## Evaluation

```bash
python eval.py --model weights/model.pth --dataset kodak \
  --root /path/to/kodak --output results/kodak.jsonl --device cuda:0
python score.py results/kodak.jsonl
```

For multi-GPU evaluation, see `scripts/evaluate_all.sh`.

