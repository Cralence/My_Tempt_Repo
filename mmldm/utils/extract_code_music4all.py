import os
import torch
from omegaconf import OmegaConf
from madmom.features.downbeats import DBNDownBeatTrackingProcessor as DownBproc
import codecs as cs
from os.path import join as pjoin
import librosa
import argparse


from mmldm.audio.audiocraft.models.builders import get_compression_model


def main(args):
    #data_dir = 'D:/local_documentation/deep_learning/ChuangGan/music4all/audios'
    #meta_dir = 'D:/local_documentation/deep_learning/ChuangGan/music4all'
    data_dir = "/nobackup/users/yiningh/yh/music4all/audios"
    meta_dir = "/nobackup/users/yiningh/yh/music4all"
    folder_name = 'music4all_codes'
    save_dir = "/nobackup/users/yiningh/yh/music4all"
    #save_dir = 'C:/Users/Mahlering/Desktop'
    #vqvae_ckpt_path = 'C:\\Users\\Mahlering\\.cache\\huggingface\\hub\\models--facebook--musicgen-small\\snapshots\\bf842007f70cf1b174daa4f42b5e45a21d59d3ec\\compression_state_dict.bin'
    vqvae_ckpt_path = '/nobackup/users/yiningh/yh/music_dance/weight/musicgen_vqvae.bin'
    #beat_tracker_ckpt_path = 'D:/local_documentation/deep_learning/ChuangGan/music_motion_diffusion/mmldm/audio/beat_detection/pretrained_models/baseline_v1'
    #beat_tracker_ckpt_path = '/nobackup/users/yiningh/yh/music_dance/music_motion_diffusion/mmldm/audio/beat_detection/pretrained_models/baseline_v1'

    device = torch.device('cuda')

    duration = 10
    sr = 32000


    # load vqvae model
    pkg = torch.load(vqvae_ckpt_path, map_location='cpu')
    cfg = OmegaConf.create(pkg['xp.cfg'])
    model = get_compression_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    model.to(device)

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

            waveform, sr = librosa.load(music_path, sr=sr)

            # convert to tensor
            waveform = torch.FloatTensor(waveform)

            # extract feature
            x = waveform[None, None, ...].to(device)
            codes, scale = model.encode(x)

            data = {
                'codes': codes.cpu()
            }

            torch.save(data, music_save_path)

            print('%d/%d, %s processed' % (i+1, len(music_data), music_id))


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



