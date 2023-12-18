from unimumo.util import instantiate_from_config
from omegaconf import OmegaConf
import argparse
import os
import torch
from os.path import join as pjoin
import soundfile as sf
import numpy as np
import codecs as cs
from unimumo.audio.audiocraft.models.builders import get_compression_model
import pandas as pd
import random
import json
from unimumo.motion.motion_process import motion_vec_to_joint
import subprocess
from unimumo.motion import skel_animation
from pytorch_lightning import seed_everything


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # about loading and saving paths
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=False,
        help="The path to save model output",
        default="./generate_results",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=False,
        default=None,
        help="The path to music motion lm model checkpoint to load",
    )
    parser.add_argument(
        "--base",
        type=str,
        required=False,
        default=None,
        help="The path to model configs to load",
    )
    parser.add_argument(
        "--music_meta_dir",
        type=str,
        required=False,
        help="The path to music metadata dir",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music4all",
    )
    parser.add_argument(
        "--motion_meta_dir",
        type=str,
        required=False,
        help="The path to motion metadata dir",
        default='/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_dance/motion_data',
    )






    parser.add_argument(
        "-gs",
        "--guidance_scale",
        type=float,
        required=False,
        default=3.0,
        help="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        required=False,
        default=10.0,
        help="Generated music/motion time",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Change this value (any integer number) will lead to a different generation result.",
    )
