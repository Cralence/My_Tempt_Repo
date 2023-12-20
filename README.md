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
```bash
# clone project   
git clone ?

# create conda environment
cd ?
conda create -n unimumo python=3.9
conda activate unimumo

# install dependencies
pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
pip install madmom==0.16.1
 ```  

### 2. Download pretrained weight
The weight of UniMuMo consists of three parts: a music VQ-VAE, a motion VQ-VAE and a music-motion LM. 
For inference, please download the unified weight that includes all three parts from here.
For data preprocessing or training, only one or two parts of them are required for each stage. 
So please download the separate weights from here.

After downloaded, please put the weights into folder `pretrained`

### 3. Run the model
```bash
python generate.py --help
  --ckpt                The path to the trained model
  
  # about conditioning
  -mu_p --music_path    The path to the music to be conditioned on
  -mo_p --motion_path   The path to the motion to be conditioned on
  -mu_d, --music_description
                        The conditional description for music
  -mo_d, --motion_description
                        The conditional description for motion
  -t, --generation_target {mu,mo,mumo,text}
                        The output format to generate, choosing from music (mu), motion (mo), joint music motion (mumo)
                        and text description (text)
                        
  # about generation settings 
  -gs, --guidance_scale 
                        Guidance scale (Large => better quality and relavancy to text; Small => better diversity)
  --temperature         Temperature for generation
  -d, --duration        Generated music/motion time, default is 10.0
  --seed                Change this value (any integer number) will lead to a different generation result
  -bs, --batch_size     Number of samples to generate for each prompt each time
  --music_meta_dir      The path to music metadata, for loading optional text prompts, default is ./data/music
  -s --save_path        The folder path to save model output
```
Conditions and generation target and be set arbitrarily, for example:
```bash
# generate music and motion without specific conditions
python generate.py --ckpt path_to_weight -t mumo

# generate music and motion with music text description
python generate.py --ckpt path_to_weight -t mumo -mu_d descriptions_for_music

# generate music conditioned on motion and text
python generate.py --ckpt path_to_weight -t mu -mu_d descriptions_for_music -mo_p path_to_motion_condition

# generate music and motion captions
python generate.py --ckpt path_to_weight -t text -mu_p path_to_music_condition -mo_p path_to_motion_condition
```

## Train the Model

### 1. Prepare the datasets
#### 1.1 Music dataset
Please refer to the website of [Music4All](https://sites.google.com/view/contact4music4all) to download the dataset. 
After downloaded, put the audio files in folder `data/music/audios`.
#### 1.2 Motion dataset
Please download [HumanML3D](https://github.com/EricGuo5513/HumanML3D), [AIST++](https://google.github.io/aistplusplus_dataset/factsfigures.html)
and [DanceDB](https://dancedb.eu/) according to their instructions. After downloaded, please put the data and metadata 
into folder `data/motion`. Note that we have provided `Mean.npy` and `Std.npy` for motion features, which is calculated 
across all three datasets. Don't overwrite it with the mean and std from HumanML3D dataset.

### 2. Preprocess the data

#### 2.1 Split vocals from music (optional)
We use [Demucs](https://github.com/facebookresearch/demucs) for splitting music and vocal.
#### 2.2 Music code extraction and beat detection
To speed up training, we use [Encodec](https://github.com/facebookresearch/audiocraft/blob/main/docs/ENCODEC.md) to 
extract all the music codes and use [drum-aware4beat](https://github.com/SunnyCYC/drum-aware4beat) to track 
the music beat before training. Please set the correct data path in `preprocessing/extract_music_code&beat.py` and run:
```bash
python preprocessing/extract_music_code&beat.py --start 0.0 --end 1.0
```
Since this process takes a long time, if you have multiple machines, you can split the work by setting `--start` and 
`--end` to specify the start and end point of each job.

### 3. Train motion VQ-VAE
Please first check the settings in `configs/train_motion_vqvae.yaml`, e.g., the paths of datasets, number of device and node.
Then run:
```bash
python train.py --stage train_vqvae --base configs/train_motion_vqvae.yaml
```
Recovering training can be achieved by appending `-r path_to_previous_checkpoint` to above command.

Reconstruction loss can be slightly reduced by fine-tuning the motion VQ-VAE with `configs/train_motion_vqvae_finetune.yaml`. 

### 4. Pair music with motion and extract motion code


### 5. Train music-motion LM

### 6. Train captioning model

### 7. Integrate the trained weights

## Citation

## Acknowledgement

 