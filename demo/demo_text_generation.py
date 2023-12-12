from mmldm.util import instantiate_from_config
from omegaconf import OmegaConf
import argparse
import os
import torch
from os.path import join as pjoin
import soundfile as sf
import numpy as np
from mmldm.audio.audiocraft.models.builders import get_compression_model
import pandas as pd
from mmldm.motion.motion_process import motion_vec_to_joint
import subprocess
from mmldm.motion import skel_animation
import random
import sys

'''
Load paired music and motion, all made to 10 seconds
'''


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
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_motion_diffusion/demo_text_generation_results",
    )

    parser.add_argument(
        "--music_code_dir",
        type=str,
        required=False,
        help="The path to music data dir",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music4all/music4all_codes"
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
        "--motion_code_dir",
        type=str,
        required=False,
        help="The path to motion data dir",
        default='/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_dance/motion_data/aligned_motion_code_enlarged_new',
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

    args = parser.parse_args()

    save_path = args.save_path
    music_save_path = pjoin(save_path, 'music')
    motion_save_path = pjoin(save_path, 'motion')
    video_save_path = pjoin(save_path, 'video')
    feature_263_save_path = pjoin(save_path, 'feature_263')
    feature_22_3_save_path = pjoin(save_path, 'feature_22_3')
    guidance_scale = args.guidance_scale
    motion_dir = args.motion_dir
    music_code_dir = args.music_code_dir
    motion_code_dir = args.motion_code_dir
    meta_dir = args.meta_dir
    duration = args.duration

    motion_mean = np.load(pjoin(motion_dir, 'Mean.npy'))
    motion_std = np.load(pjoin(motion_dir, 'Std.npy'))
    kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                       [9, 13, 16, 18, 20]]

    music_id_list = []
    with open(pjoin(meta_dir, 'music4all_test_new.txt'), 'r') as f:
        for line in f.readlines():
            music_id_list.append(line.strip())

    motion_data_full = os.listdir(motion_code_dir)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(music_save_path, exist_ok=True)
    os.makedirs(motion_save_path, exist_ok=True)
    os.makedirs(video_save_path, exist_ok=True)
    os.makedirs(feature_263_save_path, exist_ok=True)
    os.makedirs(feature_22_3_save_path, exist_ok=True)

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

    count = 0
    with open(pjoin(save_path, f'gen_captions.txt'), 'w') as f:
        while count < len(music_id_list):
            music_id = music_id_list[count]
            motion_selection = [s for s in motion_data_full if s.split('_!humanml3d_test!_')[0] == music_id]
            motion_id = random.choice(motion_selection)

            music_code = torch.load(pjoin(music_code_dir, f'{music_id}.pth'))['codes']  # 1, 4, T
            motion_code = torch.load(pjoin(motion_code_dir, motion_id))
            motion_code = motion_code[None, ...]  # 1, 4, T

            print(f'music_id: {music_id}, motion_id: {motion_id}')

            with torch.no_grad():
                batch = {
                    'text': [''],
                    'music_code': music_code.to(device),
                    'motion_code': motion_code.to(device)
                }

                print(f'music codes: {music_code.shape}, motion codes: {motion_code.shape}')
                captions, _, _ = model.generate_captions(batch)

                # only log one each time for checking
                waveform_gen = music_vqvae.decode(music_code.to(device))
                waveform_gen = waveform_gen.cpu().squeeze()
                motion_gen = motion_vqvae.decode_from_code(music_code.to(device), motion_code.to(device))
                joint_gen = motion_vec_to_joint(motion_gen, motion_mean, motion_std)[0]
                feature_263 = motion_gen.cpu().squeeze().numpy()
                feature_263 = feature_263 * motion_std + motion_mean

            os.makedirs(save_path, exist_ok=True)

            music_filename = "%s.mp3" % music_id
            music_path = os.path.join(music_save_path, music_filename)
            try:
                sf.write(music_path, waveform_gen, 32000)
            except:
                print(f'{music_filename} cannot be saved.')

            motion_filename = "%s.mp4" % music_id
            motion_path = pjoin(motion_save_path, motion_filename)
            try:
                skel_animation.plot_3d_motion(
                    motion_path, kinematic_chain, joint_gen, title='None', vbeat=None,
                    fps=20, radius=4
                )
            except:
                print(f'{motion_filename} cannot be saved.')

            video_filename = "%s.mp4" % music_id
            video_path = pjoin(video_save_path, video_filename)
            try:
                subprocess.call(
                    f"ffmpeg -i {motion_path} -i {music_path} -c copy {video_path}",
                    shell=True)
            except:
                print(f'{video_path} cannot be saved.')

            feature_263_filename = "%s.npy" % music_id
            feature_263_path = pjoin(feature_263_save_path, feature_263_filename)
            np.save(feature_263_path, feature_263)

            feature_22_3_filename = "%s.npy" % music_id
            feature_22_3_path = pjoin(feature_22_3_save_path, feature_22_3_filename)
            np.save(feature_22_3_path, joint_gen)

            description = captions[0]
            f.write(f'{music_id}\t{description}\n')
            print(f'{music_id}\t{description}', file=sys.stderr)

            count += 1
