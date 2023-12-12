import os
import torch
from omegaconf import OmegaConf
from madmom.features.downbeats import DBNDownBeatTrackingProcessor as DownBproc
import codecs as cs
from os.path import join as pjoin
import numpy as np
import librosa
import argparse
import soundfile as sf


from unimumo.audio.audiocraft.models.builders import get_compression_model
from unimumo.audio.beat_detection.track_beat import get_music_beat, build_beat_tracker


def main(args):
    data_dir = 'D:/local_documentation/deep_learning/ChuangGan/music4all/audios'
    meta_dir = 'D:/local_documentation/deep_learning/ChuangGan/music4all'
    #data_dir = "/nobackup/users/yiningh/yh/music4all/audios"
    #meta_dir = "/nobackup/users/yiningh/yh/music4all"
    folder_name = 'music4all_feature'
    #save_dir = "/nobackup/users/yiningh/yh/music4all"
    save_dir = 'C:/Users/Mahlering/Desktop'
    vqvae_ckpt_path = 'C:\\Users\\Mahlering\\.cache\\huggingface\\hub\\models--facebook--musicgen-small\\snapshots\\bf842007f70cf1b174daa4f42b5e45a21d59d3ec\\compression_state_dict.bin'
    #vqvae_ckpt_path = '/nobackup/users/yiningh/yh/music_dance/weight/musicgen_vqvae.bin'
    beat_tracker_ckpt_path = 'D:/local_documentation/deep_learning/ChuangGan/music_motion_diffusion/mmldm/audio/beat_detection/pretrained_models/baseline_v1'
    #beat_tracker_ckpt_path = '/nobackup/users/yiningh/yh/music_dance/music_motion_diffusion/unimumo/audio/beat_detection/pretrained_models/baseline_v1'

    device = torch.device('cuda')

    beat_tracker_config = {
              'model_type': 'bsl_blstm',
              #model_dir: './unimumo/audio/beat_detection/pretrained_models/baseline_v1'
              'model_dir': '/nobackup/users/yiningh/yh/music_dance/music_motion_diffusion/unimumo/audio/beat_detection/pretrained_models/baseline_v1',
              'model_simpname': 'baseline_v1',
              'num_tempi': 60,
              'transition_lambda': 140,
              'observation_lambda': 8,
              'threshold': 0.55,
              'f_measure_threshold': 0.07,
              'beats_per_bar': [ 3, 4 ],
              'max_bpm': 215,
              'min_bpm': 55,
              'fps': 100,
    }

    duration = 10
    sr = 32000


    # load vqvae model
    pkg = torch.load(vqvae_ckpt_path, map_location='cpu')
    cfg = OmegaConf.create(pkg['xp.cfg'])
    model = get_compression_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    model.to(device)

    # load beat detection model
    beat_detection_model = build_beat_tracker(beat_tracker_ckpt_path)
    hmm_proc = DownBproc(**beat_tracker_config)
    music_target_length = int(duration * sr)

    # prepare for data
    music_data = []
    music_ignore = []
    with cs.open(pjoin(meta_dir, 'music4all_ignore.txt'), "r") as f:
        for line in f.readlines():
            music_ignore.append(line.strip())
    for split in ['train', 'test', 'val']:
        with cs.open(pjoin(meta_dir, f'music4all_{split}.txt'), "r") as f:
            for line in f.readlines():
                if line.strip() in music_ignore:
                    continue
                music_data.append(line.strip())

    # prepare for save dir
    feature_dir = pjoin(save_dir, folder_name)
    os.makedirs(feature_dir, exist_ok=True)

    music_data.sort()
    start_idx = int(args.start * len(music_data))

    if args.reverse:
        music_data = music_data[:start_idx]
        music_data.reverse()
    else:
        music_data = music_data[start_idx:]

    with torch.no_grad():
        # traverse the data
        for i, music_id in enumerate(music_data):
            music_save_path = pjoin(feature_dir, music_id + '.pth')
            if os.path.exists(music_save_path):
                print('%s.pth exists' % music_id)
                continue

            if os.path.exists(pjoin(data_dir, f'{music_id}.mp3')):
                music_path = pjoin(data_dir, f'{music_id}.mp3')
            elif os.path.exists(pjoin(data_dir, f'{music_id}.wav')):
                music_path = pjoin(data_dir, f'{music_id}.wav')
            else:
                continue

            beat, bpm = get_music_beat(music_pth=music_path,
                                       rnn=beat_detection_model,
                                       hmm_proc=hmm_proc,
                                       device=device)
            if len(beat) == 0:
                print('music ' + str(music_id) + ' have failed beat detection with len 0')
                continue

            beat = (beat * sr).astype(int)

            if beat[-1] - beat[0] < music_target_length:
                print('music ' + str(music_id) + ' have failed beat detection')
                continue

            waveform, sr = librosa.load(music_path, sr=sr)

            # convert to tensor
            beat = torch.tensor(beat, dtype=torch.int)
            waveform = torch.FloatTensor(waveform)

            # extract feature
            x = waveform[None, None, ...].to(device)
            x, scale = model.preprocess(x)
            emb = model.encoder(x)
            q_res = model.quantizer(emb, model.frame_rate)

            music_token = q_res.x

            data = {
                'music_token': music_token.cpu(),  # (1, dim, n)
                'beat': beat,  # (n)
            }

            torch.save(data, music_save_path)

            # convert mp3 to wav
            os.system('rm %s' % music_path)
            music_path = pjoin(data_dir, f'{music_id}.wav')
            sf.write(music_path, waveform.cpu().squeeze().numpy(), samplerate=sr)

            print('%d/%d, %s processed, bpm: %d' % (i+1, len(music_data), music_id, bpm))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-r',
        '--reverse',
        type=bool,
        required=False,
        default=False
    )

    parser.add_argument(
        '-s',
        '--start',
        type=float,
        required=False,
        default=0
    )

    args = parser.parse_args()

    main(args)



