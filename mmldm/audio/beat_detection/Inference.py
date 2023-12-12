# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:16:50 2020

@author: CITI
"""


#%%

import os
import torch
from pathlib import Path
import numpy as np
import soundfile

import pandas as pd
from scipy.special import softmax

from models.BaselineBLSTM import RNNDownBeatProc as bsl_blstm
from models.DrumAwareBeatTracker1 import DrumAwareBeatTracker as DA1
from models.DrumAwareBeatTracker2 import DrumAwareBeatTracker as DA2

from madmom.features.downbeats import DBNDownBeatTrackingProcessor as DownBproc
from madmom.features.downbeats import RNNDownBeatProcessor as RNNproc_api

import librosa
# from da_utils import *
import da_utils as utils
#import mir_eval

f_measure_threshold=0.07 # 70ms tolerance as set in paper
beats_per_bar = [3, 4]

#%%
def main():
    cuda_num = 0 
    cuda_str = 'cuda:'+str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

    # creating inputinfo_list for evaluation

    modelinfo = {'model_type': 'bsl_blstm',
                 'model_dir': './pretrained_models/baseline_v1',
                 'model_simpname': 'baseline_v1',
                 'model_setting': float('nan'),
                 'n_tempi': 60,
                 'transition_lambda': 140,
                 'observation_lambda': 8,
                 'threshold': 0.55
                 }

    audio_file_path = './music.mp3'
        # break
        ### RNN init

    rnn = bsl_blstm()

    model_fn = 'RNNBeatProc.pth'
    model_path = os.path.join(modelinfo['model_dir'] , model_fn)

    state = torch.load(model_path, map_location = device)
    rnn.load_state_dict(state)
#            rnn.cuda(device.index)

    ### DBN init
    ### can adjust HMM tempo range here
    max_bpm = 215 # default
    min_bpm = 55 # defaul

    hmm_proc = DownBproc(beats_per_bar = beats_per_bar, min_bpm = min_bpm,
                         max_bpm = max_bpm, num_tempi = modelinfo['n_tempi'],
                         transition_lambda = modelinfo['transition_lambda'],
                         observation_lambda = modelinfo['observation_lambda'],
                         threshold = modelinfo['threshold'], fps = 100)
    ### get feature of input audio file
    feat = utils.get_feature(audio_file_path)


    activation = utils.get_dlm_activation(rnn, device, feat)

    ### Process activation into beat estimation
    if type(activation) ==list:
        if len(activation) == 4:
        ### For DA models :
            (fuser_activation, mix_activation, nodrum_activation, drum_activation) = activation
            beat_fuser_est = hmm_proc( fuser_activation)
            beat_mix_est = hmm_proc( mix_activation)
            beat_nodrum_est = hmm_proc( nodrum_activation)
            beat_drum_est = hmm_proc( drum_activation)
        else:
            print("unexpected len of activation")
    else:
        fuser_activation = activation
        beat_fuser_est = hmm_proc( fuser_activation)
        beat_mix_est = [] # not implemented for bsl model
        beat_nodrum_est = [] # not implemented for bsl model
        beat_drum_est = [] # not implemented for bsl model

    ### save fuser results
    txt_out_folder = os.path.join('./inference/out_txt', modelinfo['model_simpname'])
    if not os.path.exists(txt_out_folder):
        Path(txt_out_folder).mkdir(parents = True, exist_ok = True)
    txt_out_path = os.path.join(txt_out_folder, os.path.basename(audio_file_path)+'.beats')
    np.savetxt(txt_out_path, beat_fuser_est, fmt = '%.5f')

    # downbeat = beat_fuser_est[np.where(beat_fuser_est[:,1]==1), 0]
    beat = beat_fuser_est[:, 0]
    ori_wav = utils.get_wav(audio_file_path)
    click = librosa.clicks(times = beat, sr = 44100, length = len(ori_wav))
    click_wav = ori_wav + click
    #librosa.output.write_wav(os.path.join(txt_out_folder, os.path.basename(audio_file_path)), click_wav, sr = 44100)
    soundfile.write(os.path.join(txt_out_folder, os.path.basename(audio_file_path)), click_wav, samplerate = 44100)


if __name__ == "__main__":
    main()
