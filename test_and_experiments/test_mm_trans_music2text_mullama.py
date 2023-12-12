import random
import librosa
import torch
from os.path import join as pjoin
import argparse
import numpy as np
import json
import os
from omegaconf import OmegaConf
import sys
from unimumo.audio.audiocraft.models.builders import get_compression_model

from unimumo.util import instantiate_from_config


def load_model_from_config(config, ckpt, verbose=False):
    model = instantiate_from_config(config.model)

    if ckpt is not None:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"]
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=False,
        help="The path to save model output",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_motion_diffusion/test_music2text_mullama_v15e47",
    )

    parser.add_argument(
        "--music_dir",
        type=str,
        required=False,
        help="The path to music data dir",
        default='/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/MU-LLaMA/data/audios'
    )

    parser.add_argument(
        "--motion_code_dir",
        type=str,
        required=False,
        help="The path to motion data dir",
        default='/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_dance/motion_data/aligned_motion_code_enlarged_new'
    )

    parser.add_argument(
        "--meta_dir",
        type=str,
        required=False,
        help="The path to meta data dir",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/MU-LLaMA/data",
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
        default=3.0,
        help="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
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

    args = parser.parse_args()

    save_path = args.save_path
    music_save_path = pjoin(save_path, 'music')
    motion_save_path = pjoin(save_path, 'motion')
    video_save_path = pjoin(save_path, 'video')
    guidance_scale = args.guidance_scale
    meta_dir = args.meta_dir
    motion_dir = args.motion_dir
    music_dir = args.music_dir
    motion_code_dir = args.motion_code_dir

    motion_mean = np.load(pjoin(motion_dir, 'Mean.npy'))
    motion_std = np.load(pjoin(motion_dir, 'Std.npy'))
    kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                       [9, 13, 16, 18, 20]]

    music_id_list = []
    music_description_list = []
    with open(pjoin(meta_dir, 'EvalMusicQA.txt'), 'r') as f:
        data_list = json.load(f)
        for data in data_list:
            if data['conversation'][0]['value'] == 'Describe the audio':
                description = data['conversation'][1]['value']
                if description == 'Describe the audio' or description == 'What is the audio about?' or description == 'What is the genre of the audio?':
                    continue
                if 'Describe the audio: ' in description:
                    description = description[len('Describe the audio: '):]

                music_id_list.append(data['audio_name'])
                music_description_list.append(data['conversation'][1]['value'])

    print(f'Total number of test music: {len(music_id_list)}', file=sys.stderr)

    motion_data_list = os.listdir(motion_code_dir)

    print(f'Number of motion data: {len(motion_data_list)}', file=sys.stderr)

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

    total_num = len(music_id_list)
    print(f'total number of test data: {total_num}', file=sys.stderr)
    count = 0

    f_gen = open(pjoin(save_path, 'gen_captions.txt'), 'w')
    f_gt = open(pjoin(save_path, 'gt_captions.txt'), 'w')

    while count < total_num:
        music_id = music_id_list[count]
        waveform, sr = librosa.load(pjoin(music_dir, music_id), sr=32000)

        # take the first 10 s of the waveform
        len_waveform = int(len(waveform) / sr)
        len_waveform = min(len_waveform, 10)
        waveform = waveform[:sr * len_waveform]

        waveform = torch.FloatTensor(waveform)[None, None, ...].to(device)
        music_code, scale = music_vqvae.encode(waveform)  # 1, 4, T

        # random choose a motion code
        motion_name = random.choice(motion_data_list)
        motion_code = torch.load(pjoin(motion_code_dir, motion_name))  # 4, T
        motion_length = len_waveform * 50
        motion_code = motion_code[:, :motion_length]
        motion_code = motion_code[None, ...].to(device)

        assert music_code.shape[-1] == motion_code.shape[-1]

        with torch.no_grad():
            print(f'music code shape: {music_code.shape}, motion code shape: {motion_code.shape}', file=sys.stderr)

            batch = {
                'music_code':  music_code,
                'motion_code': motion_code,
                'text': ['']
            }

            captions, _, _ = model.generate_captions(batch)

        description = captions[0]
        description = description.split('.')
        description = [d for d in description if not (' dance ' in d or ' motion ' in d or ' dance' in d or ' motion' in d)]
        if len(description) == 0:
            description = captions[0].split('.')[0] + '.'

        description = '.'.join(description)

        print(f'Generated caption: {description}', file=sys.stderr)

        f_gen.write(f'{music_id}\t{description}\n')
        f_gt.write(f'{music_id}\t{music_description_list[count]}\n')

        count += 1

    f_gt.close()
    f_gen.close()









