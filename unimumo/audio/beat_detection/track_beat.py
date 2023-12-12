import os
import numpy as np
import soundfile
import matplotlib.pyplot as plt
from madmom.features.downbeats import DBNDownBeatTrackingProcessor as DownBproc
import random
import torch

from unimumo.audio.beat_detection.models.BaselineBLSTM import RNNDownBeatProc as bsl_blstm

import librosa
import unimumo.audio.beat_detection.da_utils as utils


def get_music_beat(music_pth, rnn, hmm_proc, device):
    feat = utils.get_feature(music_pth)
    activation = utils.get_dlm_activation(rnn, device, feat)
    beat_fuser_est = hmm_proc(activation)
    beat = beat_fuser_est[:, 0]

    if len(beat) > 0:
        bpm = (len(beat) - 1) / (beat[-1] - beat[0]) * 60
    else:
        bpm = 0

    '''    
    if 0 < bpm < 107:
        # double the bpm
        inserted_beat = list(beat)
        orinigal_len = len(inserted_beat)
        for i in range(orinigal_len - 1):
            inserted_beat.append((inserted_beat[i] + inserted_beat[i + 1]) / 2)
        inserted_beat.sort()
        beat = np.array(inserted_beat)
        bpm = (len(beat) - 1) / (beat[-1] - beat[0]) * 60'''

    return beat, bpm


def build_beat_tracker(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rnn = bsl_blstm()

    model_fn = 'RNNBeatProc.pth'
    model_path = os.path.join(model_dir, model_fn)
    state = torch.load(model_path, map_location=device)
    rnn.load_state_dict(state)

    return rnn


def random_cut_wav(waveform, segment_length, start, end):
    waveform_length = waveform.shape[-1]
    max_cut_length = end - start
    assert waveform_length > 100, "Waveform is too short, %s" % waveform_length
    assert max_cut_length > segment_length, "segment is too long"

    if max_cut_length == segment_length:
        return waveform[:, start: end], start, end
    else:  # segment_length < max_cut_length
        start_idx = random.randint(start, end - segment_length)
        return waveform[:, start_idx: start_idx + segment_length], start_idx, start_idx + segment_length



if __name__ == "__main__":
    modelinfo = {'model_type': 'bsl_blstm',
                 'model_dir': './pretrained_models/baseline_v1',
                 'model_simpname': 'baseline_v1',
                 'model_setting': float('nan'),
                 'num_tempi': 60,
                 'transition_lambda': 140,
                 'observation_lambda': 8,
                 'threshold': 0.55,
                 'f_measure_threshold': 0.07,
                 'beats_per_bar': [3, 4],
                 'max_bpm': 215,
                 'min_bpm': 55,
                 'fps': 100,
                 }
    rnn = build_beat_tracker(modelinfo['model_dir'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hmm_proc = DownBproc(**modelinfo)

    #pth = '../../../../../local_documentation/deep_learning/ChuangGan/'
    #audio_file_path = os.path.join(pth, '000/000669.wav')
    audio_file_path = './BSufNdhhpiYFbK1D.wav'

    beat, bpm = get_music_beat(music_pth=audio_file_path, rnn=rnn, hmm_proc=hmm_proc, device=device)

    ori_wav = utils.get_wav(audio_file_path)
    click = librosa.clicks(times=beat, sr=44100, length=len(ori_wav))
    click_wav = ori_wav + click
    soundfile.write('marked.wav', click_wav, samplerate=44100)

    beat = (beat * 44100).astype(int)
    beat -= int(44100 * 0.05)
    plt.figure(figsize=(30, 4))
    plt.vlines(beat, -1.0, 1.0)
    plt.vlines(beat, -1, 1, colors='red')
    plt.plot(ori_wav)
    plt.show()
