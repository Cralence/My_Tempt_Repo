from unimumo.util import instantiate_from_config
from omegaconf import OmegaConf
import argparse
import os
import torch
from os.path import join as pjoin
import soundfile as sf
import numpy as np
import codecs as cs
from unimumo.audio.audiocraft_.models.builders import get_compression_model
import pandas as pd
import random
import json
from unimumo.motion.motion_process import motion_vec_to_joint
import subprocess
from unimumo.motion import skel_animation
from pytorch_lightning import seed_everything

# use prompts from mu-llama instead of tags


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
        default="./demo_text2musicmotion",
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
        default=3,
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
        "--dropout",
        type=float,
        required=False,
        default=0.7,
        help="drop out probability",
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
    music_dir = args.music_dir
    motion_dir = args.motion_dir
    duration = args.duration
    dropout_prob = args.dropout

    seed_everything(args.seed)

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
                text_prompt.append(text_prompt_full[batch_idx])
                music_id.append(music_id_full[batch_idx])
        if len(text_prompt) == 0:
            print(f'{count}-{count + args.batch_size} exists!')
            count += args.batch_size
            continue
        print(f'generating {len(text_prompt)} audio')

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
            feature_263 = motion_gen.cpu().numpy()
            feature_263 = feature_263 * motion_std + motion_mean
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!, joint_gen: {joint_gen.shape}, waveform_gen: {waveform_gen.shape}')

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

                feature_263_filename = "%s.npy" % music_id[batch_idx]
                feature_263_path = pjoin(feature_263_save_path, feature_263_filename)
                np.save(feature_263_path, feature_263[batch_idx])

                feature_22_3_filename = "%s.npy" % music_id[batch_idx]
                feature_22_3_path = pjoin(feature_22_3_save_path, feature_22_3_filename)
                np.save(feature_22_3_path, joint_gen[batch_idx])

        count += args.batch_size

    with cs.open(pjoin(save_path, 'text_prompt.txt'), 'w') as f:
        for i in range(total_num):
            music_id = music_id_list[i]
            if os.path.exists(pjoin(music_save_path, f'{music_id}.mp3')):
                f.write(f'{music_id}\t{text_prompt_list[i]}\n')
