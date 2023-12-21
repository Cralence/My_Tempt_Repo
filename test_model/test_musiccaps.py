from omegaconf import OmegaConf
import argparse
import os
import torch
from os.path import join as pjoin
import soundfile as sf
import numpy as np
import pandas as pd
import subprocess
import random
from pytorch_lightning import seed_everything

import sys
from pathlib import Path
# Get the directory of the current script
current_dir = Path(__file__).parent
# Get the parent directory
parent_dir = current_dir.parent
# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

from unimumo.util import load_model_from_config
from unimumo.audio.audiocraft_.models.builders import get_compression_model
from unimumo.motion.motion_process import motion_vec_to_joint
from unimumo.motion import skel_animation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=False,
        help="The path to save model output",
        default="./test_result_on_musiccap",
    )

    parser.add_argument(
        "--music_dir",
        type=str,
        required=False,
        help="The path to music data dir",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music4all/audios"
    )

    parser.add_argument(
        "--meta_dir",
        type=str,
        required=False,
        help="The path to meta data dir",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music4all",
    )

    parser.add_argument(
        "--motion_dir",
        type=str,
        required=False,
        help="The path to motion data dir",
        default='/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_dance/motion_data',
    )

    parser.add_argument(
        "-gs",
        "--guidance_scale",
        type=float,
        required=False,
        default=2.5,
        help="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
    )

    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        required=False,
        default=10,
        help="Generated audio time",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Change this value (any integer number) will lead to a different generation result.",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=False,
        default=None,
        help="load checkpoint",
    )

    parser.add_argument(
        "--base",
        type=str,
        required=False,
        default=None,
        help="yaml dir",
    )

    parser.add_argument(
        "--vqvae",
        type=str,
        required=False,
        default='/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_dance/weight/musicgen_vqvae.bin',
        help="load checkpoint of pretrained vqvae",
    )

    parser.add_argument(
        "--motion_vqvae_ckpt",
        type=str,
        required=False,
        default='/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_dance/weight/mm_vqvae.ckpt',
        help="load checkpoint of pretrained motion vqvae",
    )

    parser.add_argument(
        "--motion_vqvae_config",
        type=str,
        required=False,
        default='/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_motion_diffusion/configs/mm_vqvae_v6.yaml',
        help="load checkpoint of pretrained motion vqvae config",
    )

    parser.add_argument(
        "--recover",
        type=int,
        required=False,
        default=0,
        help="recover from index",
    )

    parser.add_argument(
        "--start",
        type=float,
        required=False,
        default=0.,
        help="start ratio",
    )

    parser.add_argument(
        "--end",
        type=float,
        required=False,
        default=1.,
        help="end ratio",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=20,
        help="batch size for inference",
    )

    parser.add_argument(
        "--aist_prob",
        type=float,
        required=False,
        default=0.8,
        help="Prob of choosing AIST style motion caption.",
    )

    args = parser.parse_args()

    seed_everything(args.seed)
    save_path = args.save_path
    music_save_path = pjoin(save_path, 'music')
    motion_save_path = pjoin(save_path, 'motion')
    video_save_path = pjoin(save_path, 'video')
    guidance_scale = args.guidance_scale
    music_dir = args.music_dir
    motion_dir = args.motion_dir
    duration = args.duration

    motion_mean = np.load(pjoin(motion_dir, 'Mean.npy'))
    motion_std = np.load(pjoin(motion_dir, 'Std.npy'))
    kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                       [9, 13, 16, 18, 20]]

    # load random motion descriptions
    humanml3d_text_dir = pjoin(motion_dir, 'humanml3d_text_description')
    descriptions = os.listdir(humanml3d_text_dir)[:500]
    humanml3d_text = []
    for desc_txt in descriptions:
        with open(pjoin(motion_dir, 'humanml3d_text_description', desc_txt), 'r', encoding='UTF-8') as f:
            texts = []
            lines = f.readlines()
            for line in lines:
                text = line.split('#')[0]
                if text[-1] == '.':
                    text = text[:-1]
                humanml3d_text.append(text)
    print(f'Loaded {len(humanml3d_text)} text prompts from humanml3d')

    aist_genres = ['break', 'pop', 'lock', 'middle hip-hop', 'house', 'waack', 'krump', 'street jazz', 'ballet jazz']


    assert os.path.exists(args.meta_dir)
    music_cap_df = pd.read_csv(pjoin(args.meta_dir, 'musiccaps-public.csv'))
    text_prompt_list = list(music_cap_df['caption'])
    music_id_list = list(music_cap_df['ytid'])
    text_prompt_list = [s + ' The genre of the dance is la style hip-hop.' for s in text_prompt_list]

    print('number of testing data:', len(text_prompt_list))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(music_save_path, exist_ok=True)
    os.makedirs(motion_save_path, exist_ok=True)
    os.makedirs(video_save_path, exist_ok=True)

    pkg = torch.load(args.vqvae, map_location='cpu')
    cfg = OmegaConf.create(pkg['xp.cfg'])
    music_vqvae = get_compression_model(cfg)
    music_vqvae.load_state_dict(pkg['best_state'])
    music_vqvae.eval()

    motion_vqvae_configs = OmegaConf.load(args.motion_vqvae_config)
    motion_vqvae = instantiate_from_config(motion_vqvae_configs.model)
    pl_sd = torch.load(args.motion_vqvae_ckpt, map_location='cpu')
    motion_vqvae.load_state_dict(pl_sd['state_dict'])
    motion_vqvae.eval()

    config = OmegaConf.load(args.base)
    model = load_model_from_config(config, args.ckpt)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    music_vqvae = music_vqvae.to(device)
    motion_vqvae = motion_vqvae.to(device)

    count = max(args.recover - args.batch_size - 5, 0)
    total_num = len(text_prompt_list)
    start_idx = int(args.start * total_num)
    end_idx = int(args.end * total_num)
    count = max(start_idx, count)
    print(f'start: {count}, end: {end_idx}')
    while count < end_idx:
        # text condition
        text_prompt_full = text_prompt_list[count:min(end_idx, count + args.batch_size)]
        music_id_full = music_id_list[count:min(end_idx, count + args.batch_size)]
        print(f'{count + 1}-{min(end_idx, count + args.batch_size)}/{total_num}', end=', ')

        # check whether each file has existed
        text_prompt = []
        music_id = []
        for batch_idx in range(len(text_prompt_full)):
            if os.path.exists(pjoin(music_save_path, f'{music_id_full[batch_idx]}.mp3')):
                continue
            else:
                music_description = text_prompt_full[batch_idx]
                motion_description = None
                if random.uniform(0, 1) < args.aist_prob:
                    # use aist style prompts
                    genre = random.choice(aist_genres)
                    motion_description = f'The style of the dance is {genre}.'
                else:
                    motion_description = random.choice(humanml3d_text)
                text_prompt.append(music_description + " " + motion_description)
                music_id.append(music_id_full[batch_idx])

        if len(text_prompt) == 0:
            print(f'{count}-{count + args.batch_size} exists!')
            count += args.batch_size
            continue

        print(f'generating {len(text_prompt)} audio')

        for p in text_prompt:
            print(len(p.split(' ')), p)

        with torch.no_grad():
            batch = {
                'text': text_prompt,
                'music_code': None,
                'motion_code': None
            }

            music_gen, motion_gen, _, _, _ = model.generate_sample(
                batch,
                duration=duration,
                conditional_guidance_scale=args.guidance_scale
            )

            waveform_gen = music_vqvae.decode(music_gen)
            waveform_gen = waveform_gen.cpu().squeeze()
            motion_gen = motion_vqvae.decode_from_code(music_gen, motion_gen)
            joint_gen = motion_vec_to_joint(motion_gen, motion_mean, motion_std)

            os.makedirs(save_path, exist_ok=True)

            for batch_idx in range(len(text_prompt)):
                music_filename = "%s.mp3" % music_id[batch_idx]
                music_path = os.path.join(music_save_path, music_filename)
                try:
                    sf.write(music_path, waveform_gen[batch_idx], 32000)
                except:
                    print(f'{music_filename} cannot be saved.')
                    continue

                motion_filename = "%s.mp4" % music_id[batch_idx]
                motion_path = pjoin(motion_save_path, motion_filename)
                try:
                    skel_animation.plot_3d_motion(
                        motion_path, kinematic_chain, joint_gen[batch_idx], title='None', vbeat=None,
                        fps=20, radius=4
                    )
                except:
                    print(f'{motion_filename} cannot be saved.')
                    continue

                video_filename = "%s.mp4" % music_id[batch_idx]
                video_path = pjoin(video_save_path, video_filename)
                try:
                    subprocess.call(
                        f"ffmpeg -i {motion_path} -i {music_path} -c copy {video_path}",
                        shell=True)
                except:
                    print(f'{video_path} cannot be saved.')
                    continue

        count += args.batch_size