import torch
import os
from os.path import join as pjoin
import codecs as cs
import numpy as np
import librosa
import random
from unimumo.alignment import visual_beat, interpolation
from unimumo.motion import motion_process
from unimumo.motion.motion_process import recover_from_ric
from dtw import *
from omegaconf import OmegaConf
from unimumo.util import instantiate_from_config
from pytorch_lightning import seed_everything
import argparse
import sys

'''
pair every motion sample in humanml3d with a music track
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


def motion_vec_to_joint(vec, motion_mean, motion_std):
    # vec: [bs, 200, 263]
    mean = torch.tensor(motion_mean).to(vec)
    std = torch.tensor(motion_std).to(vec)
    vec = vec * std + mean
    joint = recover_from_ric(vec, joints_num=22)
    joint = joint.cpu().detach().numpy()
    return joint


parser = argparse.ArgumentParser()

parser.add_argument(
    '-r',
    "--reverse",
    type=bool,
    required=False,
    default=False
)

parser.add_argument(
    '-s',
    "--start",
    type=float,
    required=False,
    default=0.0
)

parser.add_argument(
    '-e',
    "--end",
    type=float,
    required=False,
    default=1.0
)

args = parser.parse_args()

music_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music4all/audios'
music_metadata_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music4all/'
motion_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_dance/motion_data'
feature_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music4all/music4all_beat'
music_code_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music4all/music4all_codes'
motion_feature_save_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_dance/motion_data/aligned_humanml3d_test_motion_code'

ckpt = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_dance/weight/mm_vqvae.ckpt'
yaml_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_motion_diffusion/configs/mm_vqvae_v6.yaml'

os.makedirs(motion_feature_save_dir, exist_ok=True)
seed_everything(2023)

music_duration = 30
max_motion_length = int(music_duration * 20)
max_music_length = int(music_duration * 32000)

config = OmegaConf.load(yaml_dir)
model = load_model_from_config(config, ckpt)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
model.eval()

ignore = []
motion_data = []
motion_mean = np.load(pjoin(motion_dir, 'Mean.npy'))
motion_std = np.load(pjoin(motion_dir, 'Std.npy'))

with cs.open(pjoin(motion_dir, 'ignore_list.txt'), "r") as f:
    for line in f.readlines():
        ignore.append(line.strip())

with cs.open(pjoin(motion_dir, f'humanml3d_test.txt'), "r") as f:
    for line in f.readlines():
        if line.strip() in ignore:
            continue
        if not os.path.exists(pjoin(motion_dir, 'test', 'joint_vecs', line.strip() + '.npy')):
            continue
        motion_data.append(line.strip())

music_data = []
for split in ['test']:
    with cs.open(pjoin(music_metadata_dir, f'music4all_{split}.txt'), "r") as f:
        for line in f.readlines():
            if not os.path.exists(pjoin(music_dir, line.strip() + '.wav')):
                continue
            if not os.path.exists(pjoin(feature_dir, line.strip() + '.pth')):
                continue
            if not os.path.exists(pjoin(music_code_dir, line.strip() + '.pth')):
                continue
            music_data.append(line.strip())

data_start_idx = int(args.start * len(motion_data))
data_end_idx = int(args.end * len(motion_data))
motion_data = motion_data[data_start_idx:data_end_idx]

print(f'total motion: {len(motion_data)}', file=sys.stderr)
print(f'total music: {len(music_data)}', file=sys.stderr)

for data_idx, motion_id in enumerate(motion_data):
    # check whether the motion has already been paired
    generated_motion_list = os.listdir(motion_feature_save_dir)
    generated_motion_list = [f for f in generated_motion_list if f.split('_!humanml3d_test!_')[1][:-4] == motion_id]
    if len(generated_motion_list) > 0:
        print(f'{data_idx + 1}/{len(motion_data)} already exists!', file=sys.stderr)
        continue
    else:
        print(f'pairing {data_idx + 1}/{len(motion_data)}!', file=sys.stderr)

    motion = np.load(pjoin(motion_dir, 'test', 'joint_vecs', motion_id + '.npy'))
    motion_length = motion.shape[0]
    # if motion length longer than 10 sec
    aug = max_motion_length // motion_length
    if aug < 1:
        start_idx = random.randint(0, motion_length - max_motion_length)
        motion = motion[start_idx:start_idx + max_motion_length]
        # length = self.max_motion_length
    elif aug == 1:
        if max_motion_length - motion_length <= 50:
            motion = motion
            # length = motion_length
        else:
            motion = np.tile(motion, (2, 1))
            # length = motion.shape[0]
    else:
        max_repeat = aug
        if max_motion_length - max_repeat * motion.shape[0] > 50:
            max_repeat += 1
        motion = np.tile(motion, (max_repeat, 1))
        # length = motion.shape[0]]

    print(f'{data_idx + 1}/{len(motion_data)}', end=' ', file=sys.stderr)

    while True:
        music_id = random.choice(music_data)
        music_path = pjoin(music_dir, f'{music_id}.wav')
        waveform, sr = librosa.load(music_path, sr=32000)

        waveform = torch.FloatTensor(waveform)
        if waveform.shape[0] != max_music_length:
            if waveform.shape[0] > max_music_length:
                waveform = waveform[:max_music_length]
            else:
                zero_pad = torch.zeros(max_music_length)
                zero_pad[:waveform.shape[0]] = waveform
                waveform = zero_pad
        waveform = waveform[None, None, ...]

        feature_dict = torch.load(pjoin(feature_dir, f'{music_id}.pth'))
        mbeat = feature_dict['beat']

        scale_ratio = 32000 / 20
        mbeat = (mbeat / scale_ratio).numpy()
        mbeat = (np.rint(mbeat)).astype(int)

        # augmented motion
        # T x 263
        try:
            rec_ric_data = motion_process.recover_from_ric(torch.from_numpy(motion).unsqueeze(0).float(), 22)
            skel = rec_ric_data.squeeze().numpy()
            directogram, vimpact = visual_beat.calc_directogram_and_kinematic_offset(skel)
            peakinds, peakvals = visual_beat.get_candid_peaks(
                vimpact, sampling_rate=20)
            tempo_bpms, result = visual_beat.getVisualTempogram(
                vimpact, window_length=4, sampling_rate=20)
            visual_beats = visual_beat.find_optimal_paths(
                list(map(lambda x, y: (x, y), peakinds, peakvals)), result, sampling_rate=20)
            vbeats = np.zeros((skel.shape[0]))
            if len(visual_beats) != 0:
                for beat in visual_beats[0]:
                    idx = beat[0]
                    vbeats[idx] = 1
        except IndexError:
            print('bad music')
            continue

        mbeats = np.zeros((max_motion_length))
        for beat in mbeat:
            if beat < len(mbeats):
                mbeats[beat] = 1

        try:
            alignment = dtw(
                vbeats, mbeats, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "d"))
            wq = warp(alignment, index_reference=False)
            final_motion = interpolation.interp(motion, wq)
            break
        except ValueError:
            print('bad', motion.shape)
            continue

    motion = (final_motion - motion_mean) / motion_std
    motion = torch.FloatTensor(motion)  # T, D

    if motion.shape[0] != max_motion_length:  # pad the motion into the same shape
        if motion.shape[0] > max_motion_length:
            motion = motion[:max_motion_length, :]
        else:
            zero_pad = torch.zeros((max_motion_length, 263))
            zero_pad[:motion.shape[0], :] = motion
            motion = zero_pad

    motion = motion[None, ...]

    zero_waveform = torch.zeros_like(waveform)

    input_batch = {'motion': motion.to(device), 'music': zero_waveform.to(device)}

    music_emb, motion_emb = model.encode(input_batch)
    #
    # q_res_music = model.quantizer(music_emb, 50)  # 50 is the fixed sample rate
    # music_code = model.quantizer.encode(music_emb)
    #
    motion_code = model.quantizer.encode(motion_emb)
    #
    # motion_quantized_representation = model.quantizer.decode(motion_code)
    # music_quantized_representation = torch.zeros_like(motion_quantized_representation).to(device)
    # motion_recon = model.decode(music_quantized_representation, motion_quantized_representation).cpu()
    #
    # curr_loss = torch.nn.functional.mse_loss(motion, motion_recon)
    # print(f'split: {split}, {data_idx}/{len(music_data)}, loss: {curr_loss}')
    #
    motion_token = motion_code.squeeze()
    print(f'motion token: {motion_token.shape}, motion: {motion.shape}, music: {zero_waveform.shape}', file=sys.stderr)
    motion_token = motion_token.cpu()

    motion_token_save_path = pjoin(motion_feature_save_dir, music_id + f'_!humanml3d_test!_{motion_id}.pth')
    print(f'save dir: {motion_token_save_path}')
    torch.save(motion_token, motion_token_save_path)  # 4, 1500




