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
        default="./test_text2motion_humanml3d",
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

    args = parser.parse_args()

    save_path = args.save_path
    music_save_path = pjoin(save_path, 'music')
    motion_save_path = pjoin(save_path, 'motion')
    video_save_path = pjoin(save_path, 'video')
    feature_save_path = pjoin(save_path, 'feature')
    joint_save_path = pjoin(save_path, 'joint')
    guidance_scale = args.guidance_scale
    music_dir = args.music_dir
    motion_dir = args.motion_dir
    duration = args.duration

    motion_mean = np.load(pjoin(motion_dir, 'Mean.npy'))
    motion_std = np.load(pjoin(motion_dir, 'Std.npy'))
    kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                       [9, 13, 16, 18, 20]]

    motion_id_list = []
    with open(pjoin(motion_dir, 'humanml3d_test.txt'), 'r') as f:
        for line in f.readlines():
            if os.path.exists(pjoin(motion_dir, 'test', 'joint_vecs', line.strip() + '.npy')):
                motion_id_list.append(line.strip())

    text_dir = pjoin(motion_dir, 'humanml3d_text_description')
    descriptions = os.listdir(text_dir)

    # the text description for humanml3d motion dataset
    humanml3d_text = {}
    for desc_txt in descriptions:
        if desc_txt.split('.')[0] in motion_id_list:
            with open(pjoin(motion_dir, 'humanml3d_text_description', desc_txt), 'r', encoding='UTF-8') as f:
                texts = []
                lines = f.readlines()
                for line in lines:
                    text = line.split('#')[0]
                    if text[-1] == '.':
                        text = text[:-1]
                    texts.append(text)
                if len(texts) == 0:
                    continue
                humanml3d_text[desc_txt.split('.')[0]] = texts

    print('number of testing data:', len(motion_id_list), len(humanml3d_text))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(music_save_path, exist_ok=True)
    os.makedirs(motion_save_path, exist_ok=True)
    os.makedirs(video_save_path, exist_ok=True)
    os.makedirs(feature_save_path, exist_ok=True)
    os.makedirs(joint_save_path, exist_ok=True)

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

    count = max(args.recover - 5, 0)
    total_num = len(motion_id_list)
    start_idx = int(args.start * total_num)
    end_idx = int(args.end * total_num)
    count = max(start_idx, count)
    print(f'start: {count}, end: {end_idx}')
    while count < end_idx:
        motion_id = motion_id_list[count]
        if os.path.exists(pjoin(feature_save_path, motion_id + '.npy')):
            count += 1
            continue
        motion_description = f'The motion is that {humanml3d_text[motion_id][0]}.'  # always take the first description
        music_description = 'The music is a rock, pop music, with fast tempo, which is intense.'
        text_prompt = music_description + ' ' + motion_description

        motion_gt = np.load(pjoin(motion_dir, 'test', 'joint_vecs', motion_id + '.npy'))
        duration = motion_gt.shape[0] / 20

        with torch.no_grad():
            batch = {
                'text': [text_prompt],
                'music_code': None,
                'motion_code': None
            }

            print(f'Generating {duration} seconds, prompt: {text_prompt}')
            music_gen, motion_gen, _, _, _ = model.generate_sample(
                batch,
                duration=duration,
                conditional_guidance_scale=args.guidance_scale
            )

            if count % 50 == 0:
                waveform_gen = music_vqvae.decode(music_gen)
                waveform_gen = waveform_gen.cpu().squeeze()
            else:
                waveform_gen = None
            motion_gen = motion_vqvae.decode_from_code(music_gen, motion_gen)
            joint_gen = motion_vec_to_joint(motion_gen, motion_mean, motion_std)[0]
            feature_gen = motion_gen.detach().cpu().squeeze().numpy()
            print(f'joint gen: {joint_gen.shape}, feature gen: {feature_gen.shape}, motion gen: {motion_gen.shape}')

            os.makedirs(save_path, exist_ok=True)

            music_filename = "%s.mp3" % motion_id
            music_path = os.path.join(music_save_path, music_filename)
            if count % 50 == 0 and waveform_gen is not None:
                try:
                    sf.write(music_path, waveform_gen, 32000)
                except:
                    print(f'{music_filename} cannot be saved.')

            motion_filename = "%s.mp4" % motion_id
            motion_path = pjoin(motion_save_path, motion_filename)
            try:
                skel_animation.plot_3d_motion(
                    motion_path, kinematic_chain, joint_gen, title='None', vbeat=None,
                    fps=20, radius=4
                )
            except:
                print(f'{motion_filename} cannot be saved.')

            video_filename = "%s.mp4" % motion_id
            video_path = pjoin(video_save_path, video_filename)
            if count % 50 == 0 and waveform_gen is not None:
                try:
                    subprocess.call(
                        f"ffmpeg -i {motion_path} -i {music_path} -c copy {video_path}",
                        shell=True)
                except:
                    print(f'{video_path} cannot be saved.')

            feature_filename = "%s.npy" % motion_id
            feature_path = pjoin(feature_save_path, feature_filename)
            np.save(feature_path, feature_gen)

            joint_filename = "%s.npy" % motion_id
            joint_path = pjoin(joint_save_path, joint_filename)
            np.save(joint_path, joint_gen)

        count += 1
