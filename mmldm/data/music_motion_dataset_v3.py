import torch
import os
import numpy as np
import codecs as cs
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from os.path import join as pjoin
import random
import pickle
import librosa
from mmldm.alignment import visual_beat, interpolation
from mmldm.motion import motion_process
from dtw import *


'''
This dataset load all training data in advance
'''

class MusicMotionDataset(Dataset):
    def __init__(self, split, music_dir, motion_dir, meta_dir, feature_dir,
                 duration=10,
                 train_humanml3d=False,
                 vqvae_sr=32000,
                 max_bpm=215,
                 music_dataset_name='fma_medium',
                 traverse_motion=True,
                 align=True,
                 music_ignore_name=None
                ):

        self.music_dir = music_dir
        self.motion_dir = motion_dir
        self.meta_dir = meta_dir
        self.feature_dir = feature_dir
        self.split = split
        self.traverse_motion = traverse_motion
        self.align = align

        self.njoints = 22
        self.fps = 20
        self.motion_dim = 263
        self.max_motion_length = duration * self.fps
        self.humanml3d = []
        self.aist = []
        self.dancedb = []
        self.motion_data = []
        self.ignore = []

        self.music_data = []
        self.music_ignore = []
        self.duration = duration
        self.vqvae_sr = vqvae_sr
        self.music_target_length = int(duration * 50)
        self.max_beat_num = int(max_bpm / 60 * duration + 20)
        if music_ignore_name is None:
            music_ignore_name = f'{music_dataset_name}_ignore.txt'

        self.bad_count = 0
        with cs.open(pjoin(self.motion_dir, 'ignore_list.txt'), "r") as f:
            for line in f.readlines():
                self.ignore.append(line.strip())
        with cs.open(pjoin(self.motion_dir, f'humanml3d_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                if not os.path.exists(pjoin(self.motion_dir, self.split, 'joint_vecs', line.strip() + '.npy')):
                    continue
                self.humanml3d.append(line.strip())
        with cs.open(pjoin(self.motion_dir, f'aist_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                if line.strip() in self.ignore:
                    continue
                if not os.path.exists(pjoin(self.motion_dir, self.split, 'joint_vecs', line.strip() + '.npy')):
                    continue
                self.motion_data.append(line.strip())
                self.aist.append(line.strip())
        with cs.open(pjoin(self.motion_dir, f'dancedb_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                if not os.path.exists(pjoin(self.motion_dir, self.split, 'joint_vecs', line.strip() + '.npy')):
                    continue
                self.motion_data.append(line.strip())
                self.dancedb.append(line.strip())

        with cs.open(pjoin(self.meta_dir, music_ignore_name), "r") as f:
            for line in f.readlines():
                self.music_ignore.append(line.strip())
        with cs.open(pjoin(self.meta_dir, f'{music_dataset_name}_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                if line.strip() in self.music_ignore:
                    continue
                if not os.path.exists(pjoin(self.music_dir, line.strip() + '.wav')):
                    continue
                if os.path.exists(pjoin(self.feature_dir, line.strip() + '.pth')):
                    self.music_data.append(line.strip())

        if not train_humanml3d:
            self.motion_data = self.motion_data*10
        else:
            self.motion_data += self.humanml3d

        self.mean = np.load(pjoin(self.motion_dir, 'Mean.npy'))
        self.std = np.load(pjoin(self.motion_dir, 'Std.npy'))
        with open(pjoin(self.motion_dir, f'{self.split}_length.pickle'), 'rb') as f:
            self.length = pickle.load(f)
        print("Total number of motions {}".format(len(self.motion_data)))

        self.loaded_motion = []
        self.loaded_music = []
        self.loaded_feature = []
        for motion_idx in tqdm(range(len(self.motion_data)), desc='Loading motion data: '):
            motion_name = self.motion_data[motion_idx]
            self.loaded_motion.append(np.load(pjoin(self.motion_dir, self.split, 'joint_vecs', motion_name + '.npy')))
        for music_idx in tqdm(range(len(self.music_data)), desc='Loading music data: '):
            music_name = self.music_data[music_idx]
            self.loaded_music.append(librosa.load(pjoin(self.music_dir, f'{music_name}.wav'), sr=self.vqvae_sr)[0])
            self.loaded_feature.append(torch.load(pjoin(self.feature_dir, f'{music_name}.pth')))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        if self.traverse_motion:
            return len(self.motion_data)
        else:
            return len(self.music_data)

    def __getitem__(self, idx):
        if self.traverse_motion:
            motion_idx = idx
            music_idx = random.randint(0, len(self.loaded_music) - 1)
        else:
            music_idx = idx
            motion_idx = random.randint(0, len(self.loaded_motion) - 1)

        motion = self.loaded_motion[motion_idx]
        motion_name = self.motion_data[motion_idx]
        if motion_name in self.humanml3d or motion_name in self.dancedb:
            motion_length = self.length[motion_name]
            # if motion length longer than 10 sec
            aug = self.max_motion_length // motion_length
            if aug < 1:
                start_idx = random.randint(0, motion_length - self.max_motion_length)
                motion = motion[start_idx:start_idx+self.max_motion_length]
                # length = self.max_motion_length
            elif aug == 1:
                if self.max_motion_length - motion_length <= 50:
                    motion = motion
                    # length = motion_length
                else:
                    motion = np.tile(motion, (2, 1))
                    # length = motion.shape[0]
            else:
                max_repeat = aug
                if self.max_motion_length - max_repeat*motion.shape[0] > 50:
                    max_repeat += 1
                motion = np.tile(motion, (max_repeat, 1))
                # length = motion.shape[0]
        else:
            motion_length = self.length[motion_name] // 3 # 60 fps -> 20 fps
            if self.max_motion_length // motion_length < 1:
                start_idx = random.randint(0, motion_length - self.max_motion_length)
                motion = motion[start_idx*3:(start_idx+self.max_motion_length)*3:3]
                # length = self.max_motion_length
            elif self.max_motion_length // motion_length == 1:
                motion = motion[::3]
                # length = motion.shape[0]
            else:
                max_repeat = self.max_motion_length // motion_length + 1
                motion = motion[::3]
                # repeat = random.randint(1, max_repeat)
                motion = np.tile(motion, (max_repeat, 1))
                # length = motion.shape[0]

        waveform = self.loaded_music[music_idx]
        waveform = waveform[None, ...]
        segment_length = int(self.music_target_length * 640)

        feature_dict = self.loaded_feature[music_idx]
        mbeat = feature_dict['beat']

        # random cut waveform and music beat
        start_idx = random.randint(0, waveform.shape[-1] - segment_length)
        end_idx = start_idx + segment_length
        waveform = waveform[:, start_idx:end_idx]
        mbeat = mbeat[torch.where((start_idx <= mbeat) & (mbeat <= end_idx))[0]]
        mbeat = mbeat - start_idx

        if self.align:
            # scale mbeat to 20 fps
            scale_ratio = self.vqvae_sr / self.fps
            mbeat = (mbeat / scale_ratio).numpy()
            mbeat = (np.rint(mbeat)).astype(int)

            # augmented motion
            # T x 263
            rec_ric_data = motion_process.recover_from_ric(torch.from_numpy(motion).unsqueeze(0).float(), self.njoints)
            skel = rec_ric_data.squeeze().numpy()
            directogram, vimpact = visual_beat.calc_directogram_and_kinematic_offset(skel)
            peakinds, peakvals = visual_beat.get_candid_peaks(
                vimpact, sampling_rate=self.fps)
            tempo_bpms, result = visual_beat.getVisualTempogram(
                vimpact, window_length=4, sampling_rate=self.fps)
            visual_beats = visual_beat.find_optimal_paths(
                list(map(lambda x, y: (x, y), peakinds, peakvals)), result, sampling_rate=self.fps)
            vbeats = np.zeros((skel.shape[0]))
            if len(visual_beats) != 0:
                for beat in visual_beats[0]:
                    idx = beat[0]
                    vbeats[idx] = 1

            mbeats = np.zeros((self.duration * self.fps))
            for beat in mbeat:
                if beat < len(mbeats):
                    mbeats[beat] = 1

            try:
                alignment = dtw(
                    vbeats, mbeats, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "d"))
                wq = warp(alignment, index_reference=False)
                final_motion = interpolation.interp(motion, wq)
            except ValueError:
                print('bad', motion.shape)
                self.bad_count += 1
                final_motion = motion
        else:
            final_motion = motion

        motion = (final_motion - self.mean) / self.std

        waveform = waveform[0, ...]
        waveform = torch.FloatTensor(waveform)

        motion = torch.FloatTensor(motion)

        if motion.shape[0] != self.max_motion_length:  # pad the motion into the same shape
            if motion.shape[0] > self.max_motion_length:
                motion = motion[:self.max_motion_length, :]
            else:
                zero_pad = torch.zeros((self.max_motion_length, self.motion_dim))
                zero_pad[:motion.shape[0], :] = motion
                motion = zero_pad

        return {
            'motion': motion,  # (200, 263)
            'waveform': waveform,  # (320000)
        }

if __name__ == "__main__":
    import time
    start = time.time()
    dataset = MusicMotionDataset(
        'val',
        music_dir='D:/local_documentation/deep_learning/ChuangGan/music4all/audios',
        feature_dir='D:/local_documentation/deep_learning/ChuangGan/music4all/music4all_feature',
        meta_dir='D:/local_documentation/deep_learning/ChuangGan/music4all',
        motion_dir='D:/local_documentation/deep_learning/ChuangGan/motion_data',
        music_dataset_name='music4all'
    )
    size = len(dataset)
    for i in range(size):
        data_dict = dataset.__getitem__(i)
        # print(data_dict['motion'].shape, data_dict['waveform'].shape, data_dict['fbank'].shape)
        if i % 100 == 0:
            end = time.time()
            print(end - start)
            start = time.time()
            if i == 100:
                break
    print(dataset.bad_count)
