# Deep Hierarchical Video Compression

This repository contains our series of works on Deep Hierarchical Video Compression.

* DHVC 1.0: The first hierarchical predictive video coding method moves away from the hybrid coding framework, achieving best-in-class performance.
*
* Paper is available at [Deep Hierarchical Video Compression (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/28733).
* 
* DHVC 2.0: will be updated soon.

### Requirments

- Python3.8+
- CUDA11.0
- pytorch1.11.0
- For others, please refer to requirements. txt

### Pretrained Models

Pretrained Models will be released soon.

### Dataset

* Train dataset: Vimeo90k
* Test dataset: UVG、MCL-JCV、HEVC Class B


### Usage

#### Testing

Please download the pretrained models into `./pretrained` and configure the environment properly mentioned above first.

```shell
python test.py
```

The testing rusults can be found in `./runs`.


### Citation

If you find this work helpful to your research, please cite

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
```

