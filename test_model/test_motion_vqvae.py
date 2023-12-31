import random
from omegaconf import OmegaConf
import argparse
import os
import torch
from os.path import join as pjoin
import numpy as np
import codecs as cs
import librosa
import soundfile as sf
from pytorch_lightning import seed_everything
from einops import rearrange

import sys
from pathlib import Path
# Get the directory of the current script
current_dir = Path(__file__).parent
# Get the parent directory
parent_dir = current_dir.parent
# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

from unimumo.util import load_model_from_config
from unimumo.motion.motion_process import motion_vec_to_joint
from unimumo.motion import skel_animation
from unimumo.motion.utils import kinematic_chain
from unimumo.audio.audiocraft_.models.builders import get_compression_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--motion_dir",
        type=str,
        required=False,
        help="The path to motion data dir",
        default='data/motion'
    )

    parser.add_argument(
        "--music_dir",
        type=str,
        required=False,
        help="The path to music data dir",
        default="data/music/audios"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        help="The path to music data dir",
        default="./test_motion_vqvae_samples"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=False,
        default='pretrained/motion_vqvae.ckpt',
        help="load checkpoint",
    )

    parser.add_argument(
        "--base",
        type=str,
        required=False,
        default='configs/train_motion_vqvae.yaml',
        help="yaml dir",
    )
    parser.add_argument(
        "--music_vqvae",
        type=str,
        required=False,
        default='pretrained/music_vqvae.bin',
        help="path to meta vqvae",
    )

    parser.add_argument(
        "--fps",
        type=int,
        required=False,
        default=20,
        choices=[20, 60],
        help="fps to load motion data",
    )

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(2023)

    motion_dir = args.motion_dir
    fps = args.fps
    mean = np.load(pjoin(motion_dir, 'Mean.npy'))
    std = np.load(pjoin(motion_dir, 'Std.npy'))

    motion_ignore_list = []
    motion_data_list = []
    aist = []
    dancedb = []

    with cs.open(pjoin(motion_dir, 'ignore_list.txt'), "r") as f:
        for line in f.readlines():
            motion_ignore_list.append(line.strip())

    with cs.open(pjoin(motion_dir, f'aist_test.txt'), "r") as f:
        for line in f.readlines():
            if line.strip() in motion_ignore_list:
                continue
            if not os.path.exists(pjoin(motion_dir, 'test', 'joint_vecs', line.strip() + '.npy')):
                continue
            motion_data_list.append(line.strip())
            aist.append(line.strip())
    with cs.open(pjoin(motion_dir, f'dancedb_test.txt'), "r") as f:
        for line in f.readlines():
            if not os.path.exists(pjoin(motion_dir, 'test', 'joint_vecs', line.strip() + '.npy')):
                continue
            motion_data_list.append(line.strip())
            dancedb.append(line.strip())

    print('number of testing data:', len(motion_data_list))

    music_dir = args.music_dir
    music_data_list = os.listdir(music_dir)

    config = OmegaConf.load(args.base)
    model = load_model_from_config(config, args.ckpt)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    pkg = torch.load(args.music_vqvae, map_location='cpu')
    cfg = OmegaConf.create(pkg['xp.cfg'])
    music_vqvae = get_compression_model(cfg)
    music_vqvae.load_state_dict(pkg['best_state'])
    music_vqvae = music_vqvae.to(device)
    music_vqvae.eval()

    count = 0
    total_num = len(motion_data_list)
    loss = 0
    loss_each_sample = []
    while count < total_num:
        motion_name = motion_data_list[count]
        motion = np.load(pjoin(motion_dir, 'test', 'joint_vecs', motion_name + '.npy'))

        if motion_name in aist and fps == 20:
            motion = motion[::3]
        if motion_name not in aist and fps == 60:
            motion = torch.Tensor(motion)
            motion = rearrange(motion, 't d -> d t')
            motion = torch.nn.functional.interpolate(motion[None, ...], scale_factor=3, mode='linear')
            motion = rearrange(motion[0], 'd t -> t d').numpy()

        motion_length = motion.shape[0]
        segment_length = fps * 10
        num_segment = motion_length // segment_length + 1

        music_name = random.choice(music_data_list)
        waveform, _ = librosa.load(pjoin(music_dir, music_name), sr=32000)

        motion = (motion - mean) / std
        motion = torch.tensor(motion)

        padded_motion = torch.zeros((num_segment * segment_length, 263))
        padded_motion[:motion_length] = motion
        motion = padded_motion[None, ...]

        music_target_length = int(num_segment * 10 * 32000)
        waveform = torch.FloatTensor(waveform)
        print(f'music shape before processed: {waveform.shape}', end=', ')
        if waveform.shape[0] < music_target_length:
            num_full_repeat = music_target_length // waveform.shape[0]
            padded_waveform = torch.zeros(music_target_length)

            for i in range(num_full_repeat):
                padded_waveform[waveform.shape[0] * i:waveform.shape[0] * (i+1)] = waveform

            padded_waveform[waveform.shape[0] * num_full_repeat:] = waveform[:music_target_length - waveform.shape[0] * num_full_repeat]
            waveform = padded_waveform
        else:
            waveform = waveform[:music_target_length]
        waveform = waveform[None, None, ...]

        print(f'after processed: {waveform.shape}')
        print(f'motion shape: {motion.shape}')

        with torch.no_grad():
            motion = motion.to(device)
            waveform = waveform.to(device)

            # waveform *= 0

            music_emb, motion_emb = model.encode(x_music=waveform, x_motion=motion)

            q_res_music = model.quantizer(music_emb, 50)  # 50 is the fixed sample rate
            q_res_motion = model.quantizer(motion_emb, 50)

            motion_recon = model.decode(q_res_music.x, q_res_motion.x)
            music_recon = music_vqvae.decoder(q_res_music.x)

            curr_loss = torch.nn.functional.mse_loss(motion, motion_recon)
            print(f'{count + 1}/{total_num}, current loss: {curr_loss},', end=' ')
            if motion_name in aist:
                print('In AIST')
            elif motion_name in dancedb:
                print('In DanceDB')
            loss += curr_loss
            loss_each_sample.append(curr_loss.item())

        if count % 20 == 0:
            joint = motion_vec_to_joint(motion_recon, mean, std)
            gt_joint = motion_vec_to_joint(motion, mean, std)

            os.makedirs(args.save_dir, exist_ok=True)

            motion_filename = f'{count}_motion_recon.mp4'
            motion_save_path = pjoin(args.save_dir, motion_filename)
            skel_animation.plot_3d_motion(
                motion_save_path, kinematic_chain, joint[0], title='None', vbeat=None,
                fps=fps, radius=4
            )

            gt_motion_filename = f'{count}_motion_gt.mp4'
            motion_save_path = pjoin(args.save_dir, gt_motion_filename)
            skel_animation.plot_3d_motion(
                motion_save_path, kinematic_chain, gt_joint[0], title='None', vbeat=None,
                fps=fps, radius=4
            )

            music_filename = f'{count}_music_recon.wav'
            music_save_path = pjoin(args.save_dir, music_filename)
            sf.write(music_save_path, music_recon.squeeze().cpu().detach().numpy(), 32000)

        count += 1

    total_loss = loss / total_num
    print(f'total loss: {total_loss}')
