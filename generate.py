import argparse
import os
import torch
import numpy as np
import random
from pytorch_lightning import seed_everything

from unimumo.models import UniMuMo
from unimumo.util import get_music_motion_prompt_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # about loading and saving paths
    parser.add_argument(
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
        help="The path to the trained model",
    )
    parser.add_argument(
        "--music_meta_dir",
        type=str,
        required=False,
        help="The path to music metadata dir",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music4all",
    )

    # about generation settings
    parser.add_argument(
        "-gs",
        "--guidance_scale",
        type=float,
        required=False,
        default=3.0,
        help="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=None,
        help="Temperature for generation",
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

    # about input prompt
    parser.add_argument(
        "-mu_p"
        "--music_path",
        type=str,
        required=False,
        default=None,
        help="The path to the music to be conditioned on",
    )
    parser.add_argument(
        " mo_p"
        "--motion_path",
        type=str,
        required=False,
        default=None,
        help="The path to the motion to be conditioned on",
    )
    parser.add_argument(
        "-mu_d"
        "--music_description",
        type=str,
        required=False,
        default=None,
        help="The conditional description of music",
    )
    parser.add_argument(
        " mo_d"
        "--motion_description",
        type=str,
        required=False,
        default=None,
        help="The conditional description of motion",
    )
    parser.add_argument(
        "-t"
        "--generation_target",
        type=str,
        required=True,
        choices=['mu', 'mo', 'mumo', 'text'],
        help="The output format to generate",
    )
    args = parser.parse_args()

    # sanity check of the arguments
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    model_ckpt = args.ckpt
    assert os.path.exists(model_ckpt)
    music_meta_dir = args.music_meta_dir
    assert os.path.exists(music_meta_dir)
    guidance_scale = args.guidance_scale
    temperature = args.temperature
    duration = args.duration
    seed_everything(args.seed)
    music_path = args.music_path
    motion_path = args.motion_path
    music_description = args.music_description
    motion_description = args.motion_description
    generation_target = args.generation_target

    # currently unconditional generation still not works well, so if description is not provided,
    # we randomly load prompts from our datasets
    music_prompt_list, motion_prompt_list = get_music_motion_prompt_list(music_meta_dir)
    if music_description is None:
        music_description = random.choice(music_prompt_list)
    if motion_description is None:
        motion_description = random.choice(motion_prompt_list)
    text_description = music_description + ' ' + motion_description

    # load model
    model = UniMuMo.from_checkpoint(model_ckpt)

    if generation_target == 'mumo':
        waveform_gen, motion_gen = model.generate_music_motion(
            text_description=text_description,
            duration=duration,
            conditional_guidance_scale=guidance_scale,
            temperature=temperature
        )

    elif generation_target == 'mu':
        assert os.path.exists(motion_path), 'When generating motion-to-music, motion path should be provided'

        motion = np.load(motion_path)
        # by default the motion is from aist, so down sample by 3
        motion = motion[::3]

        waveform_gen = model.generate_music_from_motion(
            motion_feature=motion,
            text_description=text_description,
            conditional_guidance_scale=guidance_scale,
            temperature=temperature
        )

    elif generation_target == 'mo':
        assert os.path.exists(music_path), 'When generation music-to-motion, music path should be provided'















