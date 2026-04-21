# Taming Hierarchical Image Coding Optimization: A Spectral Regularization Perspective (ICLR 2026)

[![Pytorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Conference](https://img.shields.io/badge/ICLR-2026-blue)](https://iclr.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **<a href='https://openreview.net/forum?id=lO6I66lweK'>Taming Hierarchical Image Coding Optimization: A Spectral Regularization Perspective | OpenReview</a>](#) > Wuyang Cong, Junqi Shi, Ming Lu, Xu Zhang, Zhan Ma  *Nanjing University* >

Official PyTorch implementation of the ICLR 2026 paper. We introduce a novel **Spectral Regularization** framework for Deep Hierarchical Image Compression (DHIC), specifically designed to overcome optimization bottlenecks in hierarchical architectures and achieved better performance.

## 💡 Abstract & Core Contributions

Hierarchical coding offers distinct advantages for learned image compression by capturing multi-scale representations, but its practical performance has been limited by cross-scale energy dispersion and spectral aliasing. 

To address this, we propose two explicit spectral regularization schemes:
1. **Intra-scale Frequency Regularization**: Encourages a smooth low-to-high frequency buildup as scales increase.
2. **Inter-scale Similarity Regularization**: Suppresses spectral aliasing across scales.

**Highlights:**

- 🚀 **Accelerated Convergence**: Speeds up the training of vanilla hierarchical models by **2.3$\times$**.
- 📈 **State-of-the-Art Performance**: Delivers an average **20.65% rate-distortion gain** over VTM-22.0.
- ⚡ **Zero Inference Overhead**: Both regularizers are applied *only during training* and impose absolute zero additional FLOPs or parameters during testing.

---

## 🛠️ Environment Configuration

### Prerequisites
- Linux (Ubuntu 20.04 / 22.04)
- Python $\ge$ 3.10
- CUDA $\ge$ 12.4
- PyTorch $\ge$ 2.6.0

### Installation
```bash
# 1. Create virtual environment
conda create -n dhic python=3.10 -y
conda activate dhic

# 2. Install dependencies
pip install -r requirements.txt
```

### Test

```
# 1. Download Pretrained Ckpt
From [https://box.nju.edu.cn/library/34f0db68-e19e-45db-a9cd-6fc08158b125/DHIC/](https://box.nju.edu.cn/f/4b4bb63fef8b4a3381b2/)

# 2. Select Test Datasets, such as Kodak, and run
python test_dhic.py
```

