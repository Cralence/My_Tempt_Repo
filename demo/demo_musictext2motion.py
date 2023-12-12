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
import codecs as cs
import json
import random

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
        default="./demo_musictext2motion",
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

    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=30,
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
    duration = 10
    music_code_dir = args.music_code_dir
    motion_dir = args.motion_dir

    motion_mean = np.load(pjoin(motion_dir, 'Mean.npy'))
    motion_std = np.load(pjoin(motion_dir, 'Std.npy'))
    kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                       [9, 13, 16, 18, 20]]

    music_ignore_list = []
    music_data_list = []

    assert os.path.exists(args.meta_dir)

    with cs.open(pjoin(args.meta_dir, f'music4all_ignore.txt'), 'r') as f:
        for line in f.readlines():
            music_ignore_list.append(line.strip())

    with cs.open(pjoin(args.meta_dir, f'music4all_test.txt'), "r") as f:
        for line in f.readlines():
            if line.strip() in music_ignore_list:
                continue
            music_data_list.append(line.strip())

    print('number of testing data:', len(music_data_list))
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

    aist_genres = ['break', 'pop', 'lock', 'middle hip-hop', 'LA style hip-hop', 'house', 'waack', 'krump', 'street jazz', 'ballet jazz']

    # Some text prompt
    text_prompt_list = []
    music_id_list = []

    music_prompt_list = []
    with open(pjoin(args.meta_dir, 'music4all_captions.json'), 'r') as caption_fd:
        music_caption = json.load(caption_fd)

    for music_id in music_data_list:
        if music_id not in music_caption.keys():
            continue
        music_id_list.append(music_id)
        music_prompt_list.append(music_caption[music_id])

    for i in range(len(music_prompt_list)):
        genre = random.choice(aist_genres)
        desc_choices = [f'The genre of the dance is {genre}.', f'The style of the dance is {genre}.',
                        f'This is a {genre} style dance.']
        dance_description = random.choice(desc_choices)
        text_prompt_list.append(music_prompt_list[i] + ' ' + dance_description)

    with cs.open(pjoin(save_path, 'text_prompt.txt'), 'w') as f:
        for i, text_prompt in enumerate(text_prompt_list):
            f.write(music_id_list[i] + '\t' + text_prompt + '\n')

    total_num = len(music_id_list)
    print(f'total number of test data: {total_num}')
    count = 0
    while count < total_num:
        text_prompt_full = text_prompt_list[count:min(total_num, count + args.batch_size)]
        music_id_full = music_id_list[count:min(total_num, count + args.batch_size)]
        print(f'{count + 1}-{min(total_num, count + args.batch_size)}/{total_num}', end=', ')

        batch_music_code = []
        target_length = duration * 50
        for music_id in music_id_full:
            music_path = pjoin(music_code_dir, music_id + '.pth')
            music_code = torch.load(music_path)['codes']  # 1, 4, T
            start_idx = random.randint(0, music_code.shape[2] - target_length)
            music_code = music_code[:, :, start_idx:start_idx + target_length]
            batch_music_code.append(music_code)

        music_codes = torch.cat(batch_music_code, dim=0)
        music_codes = music_codes.to(device)
        assert music_codes.shape[1] == 4 and music_codes.dim() == 3, f'{music_codes.shape}'

        with torch.no_grad():

            motion_gen = model.generate_single_modality(
                music_code=music_codes,
                motion_code=None,
                text_description=text_prompt_full,
                conditional_guidance_scale=args.guidance_scale
            )

            waveform_gen = music_vqvae.decode(music_codes).cpu().squeeze().numpy()
            motion_gen = motion_vqvae.decode_from_code(music_codes, motion_gen)
            joint_gen = motion_vec_to_joint(motion_gen, motion_mean, motion_std)
            feature_263 = motion_gen.cpu().numpy()
            feature_263 = feature_263 * motion_std + motion_mean
            print(f'feature_263: {feature_263.shape}, joint: {joint_gen.shape}')

        os.makedirs(save_path, exist_ok=True)

        for batch_idx in range(len(text_prompt_full)):
            music_filename = "%s.mp3" % music_id_full[batch_idx]
            music_path = os.path.join(music_save_path, music_filename)
            try:
                sf.write(music_path, waveform_gen[batch_idx], 32000)
            except:
                print(f'{music_filename} cannot be saved.')
                continue

            motion_filename = "%s.mp4" % music_id_full[batch_idx]
            motion_path = pjoin(motion_save_path, motion_filename)
            try:
                skel_animation.plot_3d_motion(
                    motion_path, kinematic_chain, joint_gen[batch_idx], title='None', vbeat=None,
                    fps=20, radius=4
                )
            except:
                print(f'{motion_filename} cannot be saved.')
                continue

            video_filename = "%s.mp4" % music_id_full[batch_idx]
            video_path = pjoin(video_save_path, video_filename)
            try:
                subprocess.call(
                    f"ffmpeg -i {motion_path} -i {music_path} -c copy {video_path}",
                    shell=True)
            except:
                print(f'{video_path} cannot be saved.')
                continue

            feature_263_filename = "%s.npy" % music_id_full[batch_idx]
            feature_263_path = pjoin(feature_263_save_path, feature_263_filename)
            np.save(feature_263_path, feature_263[batch_idx])

            feature_22_3_filename = "%s.npy" % music_id_full[batch_idx]
            feature_22_3_path = pjoin(feature_22_3_save_path, feature_22_3_filename)
            np.save(feature_22_3_path, joint_gen[batch_idx])

        count += args.batch_size
