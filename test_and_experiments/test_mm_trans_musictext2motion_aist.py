import random

from mmldm.util import instantiate_from_config
from omegaconf import OmegaConf
import argparse
import os
import torch
from os.path import join as pjoin
import soundfile as sf
import librosa
import numpy as np
from mmldm.audio.audiocraft.models.builders import get_compression_model
from mmldm.motion.motion_process import motion_vec_to_joint
import subprocess
from mmldm.motion import skel_animation


# generate some random motion captions for generation


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
        default="./test_music2motion",
    )

    parser.add_argument(
        "--music_dir",
        type=str,
        required=False,
        help="The path to music data dir",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/aist_plusplus_final"
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
    music_dir = args.music_dir
    motion_dir = args.motion_dir

    motion_mean = np.load(pjoin(motion_dir, 'Mean.npy'))
    motion_std = np.load(pjoin(motion_dir, 'Std.npy'))
    kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                       [9, 13, 16, 18, 20]]

    music_id_list = []
    with open(pjoin(music_dir, 'aist_audio_test_segment.txt'), 'r') as f:
        for line in f.readlines():
            music_id_list.append(line.strip())

    print('number of testing data:', len(music_id_list))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(music_save_path, exist_ok=True)
    os.makedirs(motion_save_path, exist_ok=True)
    os.makedirs(video_save_path, exist_ok=True)

    # load random motion descriptions
    humanml3d_text_dir = pjoin(motion_dir, 'humanml3d_text_description')
    descriptions = os.listdir(humanml3d_text_dir)[:100]
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

    aist_genre_map = {
        'gBR': 'break',
        'gPO': 'pop',
        'gLO': 'lock',
        'gMH': 'middle hip-hop',
        'gLH': 'LA style hip-hop',
        'gHO': 'house',
        'gWA': 'waack',
        'gKR': 'krump',
        'gJS': 'street jazz',
        'gJB': 'ballet jazz'
    }

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
    print(f'total number of test data: {total_num}')
    count = 0
    while count < total_num:
        music_path = pjoin(music_dir, music_id_list[count][1:])
        waveform, _ = librosa.load(music_path, sr=32000)
        waveform = torch.FloatTensor(waveform)
        waveform = waveform[None, None, ...].to(device)  # (1, 1, T)
        assert waveform.dim() == 3

        music_description = 'This is a pop dance music, with fast tempo and strong intensity.'
        # generate some random motion captions

        genre_id = music_id_list[count].split('/')[-1].split('_')[0]
        genre = aist_genre_map[genre_id]

        motion_description = f'The style of the dance is {genre}.'
        text_description = music_description + ' ' + motion_description

        with torch.no_grad():
            music_codes, _ = music_vqvae.encode(waveform)
            print(f'music shape: {waveform.shape}, music code shape: {music_codes.shape}', end=' ')
            print(text_description)

            motion_gen = model.generate_single_modality(
                music_code=music_codes,
                motion_code=None,
                text_description=[text_description],
                conditional_guidance_scale=args.guidance_scale
            )

            waveform_gen = music_vqvae.decode(music_codes).cpu().squeeze().numpy()
            motion_gen = motion_vqvae.decode_from_code(music_codes, motion_gen)
            joint_gen = motion_vec_to_joint(motion_gen, motion_mean, motion_std)[0]

        os.makedirs(save_path, exist_ok=True)
        music_id = music_id_list[count].split('/')[-1].split('.')[0]
        print(music_id)

        music_filename = "%s.mp3" % music_id
        music_path = os.path.join(music_save_path, music_filename)
        try:
            sf.write(music_path, waveform_gen, 32000)
        except:
            print(f'{music_filename} cannot be saved.')
            count += 1
            continue

        motion_filename = "%s.mp4" % music_id
        motion_path = pjoin(motion_save_path, motion_filename)
        try:
            skel_animation.plot_3d_motion(
                motion_path, kinematic_chain, joint_gen, title='None', vbeat=None,
                fps=20, radius=4
            )
        except:
            print(f'{motion_filename} cannot be saved.')
            count += 1
            continue

        video_filename = "%s.mp4" % music_id
        video_path = pjoin(video_save_path, video_filename)
        try:
            subprocess.call(
                f"ffmpeg -i {motion_path} -i {music_path} -c copy {video_path}",
                shell=True)
        except:
            print(f'{video_path} cannot be saved.')
            count += 1
            continue

        count += 1
