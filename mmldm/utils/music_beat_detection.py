# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:16:50 2020

@author: CITI
"""


#%%

import os
import torch
import numpy as np
import soundfile
import matplotlib.pyplot as plt

from mmldm.audio.beat_detection import RNNDownBeatProc as bsl_blstm

from madmom.features.downbeats import DBNDownBeatTrackingProcessor as DownBproc

import librosa
# from da_utils import *
import mmldm.audio.beat_detection.da_utils as utils
#import mir_eval


pth = '../../../../../local_documentation/deep_learning/ChuangGan/'
audio_file_path = os.path.join(pth, '000/000194.wav')
#audio_file_path = './music.mp3'


f_measure_threshold=0.07 # 70ms tolerance as set in paper
beats_per_bar = [3, 4]

cuda_num = 0
cuda_str = 'cuda:'+str(cuda_num)
device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

# creating inputinfo_list for evaluation

modelinfo = {'model_type': 'bsl_blstm',
             'model_dir': './mmldm/beat_detection/pretrained_models/baseline_v1',
             'model_simpname': 'baseline_v1',
             'model_setting': float('nan'),
             'n_tempi': 60,
             'transition_lambda': 140,
             'observation_lambda': 8,
             'threshold': 0.55
             }


rnn = bsl_blstm()

model_fn = 'RNNBeatProc.pth'
model_path = os.path.join(modelinfo['model_dir'] , model_fn)

state = torch.load(model_path, map_location = device)
rnn.load_state_dict(state)

### DBN init
### can adjust HMM tempo range here
max_bpm = 215 # default 215
min_bpm = 55 # defaul 55

hmm_proc = DownBproc(beats_per_bar = beats_per_bar, min_bpm = min_bpm,
                     max_bpm = max_bpm, num_tempi = modelinfo['n_tempi'],
                     transition_lambda = modelinfo['transition_lambda'],
                     observation_lambda = modelinfo['observation_lambda'],
                     threshold = modelinfo['threshold'], fps = 100)
### get feature of input audio file
feat = utils.get_feature(audio_file_path)

activation = utils.get_dlm_activation(rnn, device, feat)

### Process activation into beat estimation
fuser_activation = activation
beat_fuser_est = hmm_proc(fuser_activation)

# downbeat = beat_fuser_est[np.where(beat_fuser_est[:,1]==1), 0]
beat = beat_fuser_est[:, 0]

bpm = (len(beat)-1) / (beat[-1] - beat[0]) * 60
if bpm < 107:
    # double the bpm
    inserted_beat = list(beat)
    orinigal_len = len(inserted_beat)
    for i in range(orinigal_len - 1):
        inserted_beat.append((inserted_beat[i] + inserted_beat[i+1]) / 2)
    inserted_beat.sort()
    beat = np.array(inserted_beat)
print('average bpm: ', bpm)


ori_wav = utils.get_wav(audio_file_path)
click = librosa.clicks(times = beat, sr = 44100, length = len(ori_wav))
click_wav = ori_wav + click
soundfile.write('marked.wav', click_wav, samplerate = 44100)


beat = (beat * 44100).astype(int)
beat -= int(44100 * 0.05)
plt.figure(figsize=(30, 4))
plt.vlines(beat, -1.0, 1.0)
plt.vlines(beat, -1, 1, colors='red')
plt.plot(ori_wav)
plt.show()


def get_music_beat(music_pth, rnn, hmm_proc, device):
    feat = utils.get_feature(music_pth)
    activation = utils.get_dlm_activation(rnn, device, feat)
    beat_fuser_est = hmm_proc(activation)
    beat = beat_fuser_est[:, 0]

    bpm = (len(beat) - 1) / (beat[-1] - beat[0]) * 60
    if bpm < 107:
        # double the bpm
        inserted_beat = list(beat)
        orinigal_len = len(inserted_beat)
        for i in range(orinigal_len - 1):
            inserted_beat.append((inserted_beat[i] + inserted_beat[i + 1]) / 2)
        inserted_beat.sort()
        beat = np.array(inserted_beat)
    print('average bpm: ', bpm)

    return beat, bpm

