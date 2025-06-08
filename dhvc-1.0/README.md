## PyTorch implementation of our DHVC 1.0 [(AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/28733)
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

Follow the command below to run testing in the dhvc-1.0 folder:
```shell
python test.py -d test_dataset_name -c checkpoint_path -p test_dataset_path -g 32 -f 96 
```
`-d` represents the name of the test dataset used in log file. `-c, -p` represent the path of the pretrained models and test dataset. `-g, -f` represent the GOP size and total frame numbers for evaluation. By default, the pretrained models will be placed in `./pretrained`, the test dataset will be placed in `./dataset`. The test results can be found in `./runs`.
