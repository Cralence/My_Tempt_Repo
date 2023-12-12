import torch
import numpy as np
import codecs as cs
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from os.path import join as pjoin
import random
import pickle
import librosa
from mmldm.audio.tools import random_pad_wav, normalize_wav, _pad_spec, get_mel_from_wav
from mmldm.audio.stft import TacotronSTFT
from mmldm.alignment import music_beat, visual_beat, interpolation
from mmldm.motion import motion_process
from dtw import *

class MusicMotionDataset(Dataset):
    def __init__(self, split, music_dir, motion_dir, meta_dir,
                 filter_length=1024,
                 hop_length=160,
                 win_length=1024,
                 n_mel_channels=64,
                 sampling_rate=16000,
                 mel_fmin=0,
                 mel_fmax=8000,
                 max_motion_length=200, duration=10, random_crop=True, train_humanml3d=False):
        self.njoints = 22
        self.fps = 20
        self.music_dir = music_dir
        self.motion_dir = motion_dir
        self.meta_dir = meta_dir
        self.split = split
        self.max_motion_length = max_motion_length
        self.humanml3d = []
        self.aist = []
        self.dancedb = []
        self.motion_data = []
        self.ignore = []

        self.music_data = []
        self.music_ignore = []
        self.duration = duration
        self.music_target_length = duration * 102.4
        self.random_crop = random_crop
        self.stft = TacotronSTFT(filter_length, hop_length, win_length, n_mel_channels,
            sampling_rate, mel_fmin, mel_fmax)
        self.bad_count = 0
        with cs.open(pjoin(self.motion_dir, 'ignore_list.txt'), "r") as f:
            for line in f.readlines():
                self.ignore.append(line.strip())
        with cs.open(pjoin(self.motion_dir, f'humanml3d_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                # self.motion_data.append(line.strip())
                self.humanml3d.append(line.strip())
        with cs.open(pjoin(self.motion_dir, f'aist_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                if line.strip() in self.ignore:
                    continue
                self.motion_data.append(line.strip())
                self.aist.append(line.strip())
        with cs.open(pjoin(self.motion_dir, f'dancedb_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                self.motion_data.append(line.strip())
                self.dancedb.append(line.strip())

        with cs.open(pjoin(self.meta_dir, 'fma_medium_ignore.txt'), "r") as f:
            for line in f.readlines():
                self.music_ignore.append(line.strip())
        with cs.open(pjoin(self.meta_dir, f'fma_medium_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                if line.strip() in self.music_ignore:
                    continue
                #self.music_data.append(line.strip().split('/')[1])
                self.music_data.append(line.strip())
        if not train_humanml3d:
            self.motion_data = self.motion_data*10
        self.mean = np.load(pjoin(self.motion_dir, 'Mean.npy'))
        self.std = np.load(pjoin(self.motion_dir, 'Std.npy'))
        with open(pjoin(self.motion_dir, f'{self.split}_length.pickle'), 'rb') as f:
            self.length = pickle.load(f)
        print("Total number of motions {}".format(len(self.motion_data)))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.motion_data)

    def __getitem__(self, idx):
        motion_name = self.motion_data[idx]
        motion = np.load(pjoin(self.motion_dir, self.split, 'joint_vecs', motion_name + '.npy'))
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

        random_music_idx = random.randint(0, len(self.music_data)-1)
        music_id = self.music_data[random_music_idx]
        music_path = pjoin(self.music_dir, f'{music_id}.wav') #mp3
        waveform, sr = librosa.load(music_path, sr=16000)
        waveform = normalize_wav(waveform)
        waveform = waveform[None, ...]
        segment_length = int(self.music_target_length * 160)
        waveform = random_pad_wav(waveform, segment_length, random_crop=self.random_crop)

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
        mbeat = music_beat.music_beat(waveform[0, :sr*self.duration], sr=sr, fps=self.fps)
        mbeats = np.zeros((self.duration * self.fps))
        for beat in mbeat:
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

        motion = (final_motion - self.mean) / self.std

        waveform = waveform[0, ...]
        waveform = torch.FloatTensor(waveform)

        fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, self.stft)

        fbank = torch.FloatTensor(fbank.T)  # 1025, 64
        # log_magnitudes_stft = torch.FloatTensor(log_magnitudes_stft.T)  # 1025, 513
        fbank = _pad_spec(fbank, int(self.music_target_length))
        # log_magnitudes_stft = _pad_spec(
        #     log_magnitudes_stft, int(self.music_target_length)
        # )
        return {
            'motion': motion,
            'waveform': waveform,
            'fbank': fbank
        }

if __name__ == "__main__":
    import time
    start = time.time()
    dataset = MusicMotionDataset('train',
                                 '/media/ksu/Elements/FMA/fma/fma_medium',
                                 '/data/motion_data')
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
