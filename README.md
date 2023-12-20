<div align="center">  

# BlaBlaBla: Unified Text, Music and Motion Generation

<a href='https://www.google.com/'><img src='https://img.shields.io/badge/Demo-Page-blue'></a> 
[![Paper](http://img.shields.io/badge/paper-arxiv.2308.12064-B31B1B.svg)](https://www.google.com/)

</div>

---

This is the official repository of **BlaBlaBla**, a unified music, motion and text generation model. 
In this repository, we present model and data processing code, as well as the model weights.

---

## Brief Introduction

some introduction

with one images

## Quick Start

### 1. Conda environment

### 2. Download pretrained weight

### 3. Run the model

## Train the Model

### 1. Prepare the datasets

### 2. Preprocess the data

### 3. Train motion VQ-VAE

### 4. Train music-motion LM

### 5. Train captioning model

### 6. Integrate the trained weights

## Citation

## Acknowledgement

## How to Run   
1. Install dependencies   
```bash
# clone project   
git clone https://github.com/Cralence/SILT.git

# create conda environment
cd SILT
conda env create -f environment.yaml
conda activate silt
pip install opencv-python
pip install omegaconf==2.3.0
 ```   

2. Download the additional non-shadow dataset from [here](https://drive.google.com/file/d/1OHDCr0j6qrSYL1iDokY1kjaMcfRPepui/view?usp=drive_link) if needed. Pretrained weights for the backbone encoders
can be downloaded from the table below. Then, set the correct path and whether to use the additional 
dataset in `configs/silt_training_config.yaml`. Note that we use the additional dataset only when training on SBU.

3. Train the model by running:
```bash
python train.py --dataset SBU --backbone PVT-b5
```

4. Test the model by running:
```bash
python infer.py --dataset SBU --ckpt path_to_weight  
```

## Dataset
Our relabeled SBU test set can be downloaded from [here](https://drive.google.com/file/d/1M5YWnOJ2GtR85WJ2uhoLC-0mT2cr-ov4/view?usp=drive_link).

## Pretrained Model
|      Model      |  Params（M)  |                                                                                 Pretrained Backbone                                                                                 |                                                SBU                                                |                                               ISTD                                                |   UCF    |
|:---------------:|:-----------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|:--------:|
| EfficientNet-B3 |    12.18    |                                                                                          -                                                                                          |                                               5.23                                                |                                               2.00                                                |   9.18   |
| EfficientNet-B7 |    67.80    |                                                                                          -                                                                                          |                                               4.62                                                |                                               1.46                                                |   7.97   |
|   ResNeXt-101   |    90.50    |                                           [weight](https://drive.google.com/file/d/18U2o7msKJexwUzYuoWf4Hp_hxM0sl6IP/view?usp=drive_link)                                           |                                               5.08                                                |                                               1.53                                                |   9.27   |
|   ConvNeXt-B    |   100.68    |                                                                                          -                                                                                          |                                               5.11                                                |                                               1.15                                                |   8.62   |
|    PVT v2-B3    |    49.42    |                                           [weight](https://drive.google.com/file/d/1xIsO5uS_Z7G5WsK_qlCCdxI4GA3sYb9Y/view?usp=drive_link)                                           |                                               4.36                                                | **[1.11](https://drive.google.com/file/d/1jT2yySs_ZxG_oyD-D5xkxeyPBc1igqpL/view?usp=drive_link)** |   7.25   |
|    PVT v2-B5    |    86.14    |                                           [weight](https://drive.google.com/file/d/1fgF8pgXEgDJ2bFFLcNUeJJvMzhdr2oOa/view?usp=drive_link)                                           | **[4.19](https://drive.google.com/file/d/1CvO6xoXdUw72xGFyhHfroi4LjGyEjBKD/view?usp=drive_link)** |                                               1.16                                                | **7.23** |

### Citation   
```
@inproceedings{yang2023silt,
  title={SILT: Shadow-aware Iterative Label Tuning for Learning to Detect Shadows from Noisy Labels},
  author={Han Yang, Tianyu Wang, Xiaowei Hu, Chi-Wing Fu},
  booktitle={IEEE International Conference on Computer Vision},
  year={2023}
}
```   