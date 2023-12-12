from unimumo.util import instantiate_from_config
from omegaconf import OmegaConf
import argparse
import os
import torch
from os.path import join as pjoin
import soundfile as sf
import librosa
import numpy as np
from unimumo.audio.audiocraft.models.builders import get_compression_model
from unimumo.motion.motion_process import motion_vec_to_joint
import subprocess
from unimumo.motion import skel_animation
import random
import json
import codecs as cs


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
        default="./demo_music2motion_single",
    )

    parser.add_argument(
        "--music_path",
        type=str,
        required=True,
        help="The path to meta data dir",
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
        default='/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_motion_diffusion/mm_transformer_logs/2023-11-16T23-46-19_mm_transformer_v15/checkpoints/last.ckpt',
        help="load checkpoint",
    )

    parser.add_argument(
        "--base",
        type=str,
        required=False,
        default=None,
        help="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_motion_diffusion/configs/mm_transformer_v15.yaml",
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
        '-d',
        "--description",
        type=str,
        required=False,
        default='The music is a strong pop music. The genre of the dance is waack',
        help="load checkpoint of pretrained motion vqvae config",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=20,
        help="batch size for inference",
    )

    parser.add_argument(
        "--duration",
        type=int,
        required=False,
        default=10,
        help="batch size for inference",
    )

    args = parser.parse_args()

    save_path = args.save_path
    music_save_path = pjoin(save_path, 'music')
    motion_save_path = pjoin(save_path, 'motion')
    video_save_path = pjoin(save_path, 'video')
    feature_263_save_path = pjoin(save_path, 'feature_263')
    feature_22_3_save_path = pjoin(save_path, 'feature_22_3')
    guidance_scale = args.guidance_scale
    music_path = args.music_path
    motion_dir = args.motion_dir
    duration = args.duration

    motion_mean = np.load(pjoin(motion_dir, 'Mean.npy'))
    motion_std = np.load(pjoin(motion_dir, 'Std.npy'))
    kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                       [9, 13, 16, 18, 20]]

    description = args.description

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
    device = torch.device("cuda:4") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    music_vqvae = music_vqvae.to(device)
    motion_vqvae = motion_vqvae.to(device)

    waveform, _ = librosa.load(music_path, sr=32000)
    target_length = duration * 32000
    waveform = waveform[:target_length]
    waveform = torch.FloatTensor(waveform)[None, None, ...]
    waveform = waveform.to(device)

    with torch.no_grad():
        music_code, _ = music_vqvae.encode(waveform)
        music_code = music_code.repeat(args.batch_size, 1, 1)
        print(f'waveform: {waveform.shape}, music_code: {music_code.shape}')

        motion_gen = model.generate_single_modality(
            music_code=music_code,
            motion_code=None,
            text_description=[description] * args.batch_size,
            conditional_guidance_scale=args.guidance_scale
        )

        waveform_gen = waveform.squeeze().cpu().numpy()
        #waveform_gen = music_vqvae.decode(music_code).squeeze().cpu().numpy()
        motion_gen = motion_vqvae.decode_from_code(music_code, motion_gen)
        joint_gen = motion_vec_to_joint(motion_gen, motion_mean, motion_std)
        feature_263 = motion_gen.cpu().numpy()
        feature_263 = feature_263 * motion_std + motion_mean
        print(f'joint_gen: {joint_gen.shape}, feature 263: {feature_263.shape}')

    os.makedirs(save_path, exist_ok=True)

    for batch_idx in range(args.batch_size):

        music_filename = "%d.mp3" % batch_idx
        music_path = os.path.join(music_save_path, music_filename)
        try:
            sf.write(music_path, waveform_gen, 32000)
        except:
            print(f'{music_filename} cannot be saved.')
            continue

        motion_filename = "%d.mp4" % batch_idx
        motion_path = pjoin(motion_save_path, motion_filename)
        try:
            skel_animation.plot_3d_motion(
                motion_path, kinematic_chain, joint_gen[batch_idx], title='None', vbeat=None,
                fps=20, radius=4
            )
        except:
            print(f'{motion_filename} cannot be saved.')
            continue

        video_filename = "%d.mp4" % batch_idx
        video_path = pjoin(video_save_path, video_filename)
        try:
            subprocess.call(
                f"ffmpeg -i {motion_path} -i {music_path} -c copy {video_path}",
                shell=True)
        except:
            print(f'{video_path} cannot be saved.')
            continue

        feature_263_filename = "%d.npy" % batch_idx
        feature_263_path = pjoin(feature_263_save_path, feature_263_filename)
        np.save(feature_263_path, feature_263[batch_idx])

        feature_22_3_filename = "%d.npy" % batch_idx
        feature_22_3_path = pjoin(feature_22_3_save_path, feature_22_3_filename)
        np.save(feature_22_3_path, joint_gen[batch_idx])

