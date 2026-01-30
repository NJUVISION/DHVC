# Deep Hierarchical Video Compression
This repository contains our series of works on Deep Hierarchical Video Compression.
* __DHVC 1.0__: The first hierarchical predictive coding method moves away from the hybrid coding framework, achieving best-in-class performance. Paper is available at [Deep Hierarchical Video Compression (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/28733).
* __DHVC 2.0__: The enhanced hierarchical predictive coding method, which integrates variable-rate intra- and inter-coding into a single model, delivering not only superior compression performance to representative methods but
 also real-time processing with a significantly smaller memory footprint on standard GPUs. Paper is available at [High-Efficiency Neural Video Compression
  via Hierarchical Predictive Learning (arxiv 2024)](https://arxiv.org/pdf/2410.02598).
* __DHIC__: A spectrally regularized hierarchical image coding method, delivering not only superior compression performance but also efficient optimization without increasing inference complexity. Paper is available at [Taming Hierarchical Image Coding Optimization: A Spectral Regularization Perspective (ICLR 2026)](https://openreview.net/pdf?id=lO6I66lweK).

## News
[2026.1.26] Excited to share that our work on intra-frame hierarchical coding (DHIC) has been accepted by ICLR! The code will be open-sourced soon. 🚀

[2025.2.12] We have reconstructed the code and uploaded the pretrained models of [DHVC 1.0](https://github.com/NJUVISION/DHVC/tree/main/dhvc-1.0).

### Citation
If you find this work helpful to your research, please cite:
```
@inproceedings{lu2024deep,
  title={Deep Hierarchical Video Compression},
  author={Lu, Ming and Duan, Zhihao and Zhu, Fengqing and Ma, Zhan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={8},
  pages={8859--8867},
  year={2024}
}

@article{lu2024high,
  title={High-Efficiency Neural Video Compression via Hierarchical Predictive Learning},
  author={Lu, Ming and Duan, Zhihao and Cong, Wuyang and Ding, Dandan and Zhu, Fengqing and Ma, Zhan},
  journal={arXiv preprint arXiv:2410.02598},
  year={2024}
}
```

### Contact
If you have any question, feel free to contact us via minglu@nju.edu.cn or congwuyang@smail.nju.edu.cn.
