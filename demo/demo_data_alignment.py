import torch
import os
from os.path import join as pjoin
import codecs as cs
import numpy as np
import librosa
import random
from mmldm.alignment import visual_beat, interpolation
from mmldm.motion import motion_process
from mmldm.motion.motion_process import recover_from_ric, motion_vec_to_joint
from dtw import *
from omegaconf import OmegaConf
from mmldm.util import instantiate_from_config
from pytorch_lightning import seed_everything
import argparse
import sys
import soundfile as sf
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
motion_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_dance/motion_data'
feature_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music4all/music4all_beat'
music_code_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music4all/music4all_codes'

save_path = './demo_data_alignment'
music_save_path = pjoin(save_path, 'music')
motion_save_path = pjoin(save_path, 'motion')
video_save_path = pjoin(save_path, 'video')
feature_263_save_path = pjoin(save_path, 'feature_263')
feature_22_3_save_path = pjoin(save_path, 'feature_22_3')
os.makedirs(save_path, exist_ok=True)
os.makedirs(music_save_path, exist_ok=True)
os.makedirs(motion_save_path, exist_ok=True)
os.makedirs(video_save_path, exist_ok=True)
os.makedirs(feature_263_save_path, exist_ok=True)
os.makedirs(feature_22_3_save_path, exist_ok=True)


motion_mean = np.load(pjoin(motion_dir, 'Mean.npy'))
motion_std = np.load(pjoin(motion_dir, 'Std.npy'))
kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                   [9, 13, 16, 18, 20]]

seed_everything(2023)

music_duration = 30
max_motion_length = int(music_duration * 20)
max_music_length = int(music_duration * 32000)

motion_data = ['gWA_sBM_cAll_d27_mWA3_ch07',
    'gWA_sBM_cAll_d27_mWA3_ch10',
    'gWA_sFM_cAll_d27_mWA2_ch21',
    'gWA_sFM_cAll_d27_mWA5_ch20',
    'gMH_sFM_cAll_d24_mMH1_ch16',
    'gKR_sBM_cAll_d30_mKR5_ch09']

music_data = os.listdir(music_dir)
music_data = [music_id.split('.')[0] for music_id in music_data]
music_beat_data = os.listdir(feature_dir)
music_beat_data = [music_id.split('.')[0] for music_id in music_beat_data]
music_data = list(set(music_data) & set(music_beat_data))

for data_idx, motion_id in enumerate(motion_data):
    for example_id in range(10):
        motion = np.load(pjoin(motion_dir, 'test', 'joint_vecs', motion_id + '.npy'))
        motion = motion[::3]
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

        while True:
            music_id = random.choice(music_data)
            music_path = pjoin(music_dir, f'{music_id}.wav')
            waveform, sr = librosa.load(music_path, sr=32000)

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
                print(f'bad motion: {motion_id}, {motion.shape}')
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
                print(f'bad motion: {motion_id}, {motion.shape}')
                continue

        motion = (final_motion - motion_mean) / motion_std
        motion = torch.FloatTensor(motion)
        joint_gen = motion_vec_to_joint(motion, motion_mean, motion_std)
        feature_263 = motion * motion_std + motion_mean
        print(f'joint_gen: {joint_gen.shape}, feature 263: {feature_263.shape}, example {example_id}', file=sys.stderr)

        music_filename = "%s_%d.mp3" % (motion_id, example_id)
        music_path = os.path.join(music_save_path, music_filename)
        sf.write(music_path, waveform, 32000)

        motion_filename = "%s_%d.mp4" % (motion_id, example_id)
        motion_path = pjoin(motion_save_path, motion_filename)
        skel_animation.plot_3d_motion(
                motion_path, kinematic_chain, joint_gen, title='None', vbeat=None,
                fps=20, radius=4
            )

        video_filename = "%s_%d.mp4" % (motion_id, example_id)
        video_path = pjoin(video_save_path, video_filename)
        subprocess.call(
                f"ffmpeg -i {motion_path} -i {music_path} -c copy {video_path}",
                shell=True)

        feature_263_filename = "%s_%d.npy" % (motion_id, example_id)
        feature_263_path = pjoin(feature_263_save_path, feature_263_filename)
        np.save(feature_263_path, feature_263)

        feature_22_3_filename = "%s_%d.npy" % (motion_id, example_id)
        feature_22_3_path = pjoin(feature_22_3_save_path, feature_22_3_filename)
        np.save(feature_22_3_path, joint_gen)





