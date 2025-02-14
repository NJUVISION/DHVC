# Deep Hierarchical Video Compression
This repository contains our series of works on Deep Hierarchical Video Compression.
* DHVC 1.0: The first hierarchical predictive coding method moves away from the hybrid coding framework, achieving best-in-class performance. Paper is available at [Deep Hierarchical Video Compression (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/28733).
* DHVC 2.0: The enhanced hierarchical predictive coding method, which integrates variable-rate intra- and inter-coding into a single model, delivering not only superior compression performance to representative methods but
 also real-time processing with a significantly smaller memory footprint on standard GPUs. Paper is available at [High-Efficiency Neural Video Compression
  via Hierarchical Predictive Learning (arxiv 2024)](https://arxiv.org/pdf/2410.02598).

## News
[2025.2.12] We have reconstructed the code and uploaded the pretrained models of DHVC 1.0.

### Requirments
- Python 3.8+
- CUDA 11.0
- pytorch 1.11.0
- For others, please refer to requirements.txt

### Pretrained Models
The pretrained models of DHVC 1.0 can be downloaded from [NJU Box](https://box.nju.edu.cn/d/cda112aa5f724b7ea865/).

### Dataset
* Train dataset: Vimeo90k
* Test dataset: UVG、MCL-JCV、HEVC Class B

### Usage
#### Testing
Please download the pretrained models and configure the environment properly first.
```shell
python test.py -d test_dataset_name -c checkpoint_path -p test_dataset_path -g 32 -f 96 
```
`-d` represents the name of the test dataset used in log file. `-c, -p` represent the path of the pretrained models and test dataset. `-g, -f` represent the GOP size and total frame numbers for evaluation. By default, the pretrained models will be placed in `./pretrained`, the test dataset will be placed in `./dataset`. The test results can be found in `./runs`.

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

@article{lu2024high,
  title={High-Efficiency Neural Video Compression via Hierarchical Predictive Learning},
  author={Lu, Ming and Duan, Zhihao and Cong, Wuyang and Ding, Dandan and Zhu, Fengqing and Ma, Zhan},
  journal={arXiv preprint arXiv:2410.02598},
  year={2024}
}
```

### Citation
If you have any question, feel free to contact us via minglu@nju.edu.cn or congwuyang@smail.nju.edu.cn.
