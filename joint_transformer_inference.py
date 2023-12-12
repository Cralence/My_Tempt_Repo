from unimumo.util import instantiate_from_config
from omegaconf import OmegaConf
import argparse
import os
import torch
from os.path import join as pjoin
import soundfile as sf
import librosa
from unimumo.audio.tools import random_pad_wav, normalize_wav
import codecs as cs
from unimumo.audio.audiocraft.models.builders import get_compression_model
import pandas as pd
import random
import pytorch_lightning as pl
import numpy as np
from unimumo.motion.motion_process import recover_from_ric
from unimumo.motion import skel_animation
import subprocess


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
    vec = vec*std + mean
    joint = recover_from_ric(vec, joints_num=22)
    joint = joint.cpu().detach().numpy()
    return joint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=False,
        help="The path to save model output",
        default="./samples",
    )

    parser.add_argument(
        "--music_dir",
        type=str,
        required=False,
        help="The path to music data dir",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music4all/audios"
    )

    parser.add_argument(
        "--meta_dir",
        type=str,
        required=False,
        help="The path to meta data dir",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music4all",
    )

    parser.add_argument(
        "--motion_dir",
        type=str,
        required=False,
        help="The path to motion data dir",
        default='/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_dance/motion_data',
    )

    parser.add_argument(
        "-gs",
        "--guidance_scale",
        type=float,
        required=False,
        default=2.5,
        help="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
    )

    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        required=False,
        default=10,
        help="Generated audio time",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Change this value (any integer number) will lead to a different generation result.",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=False,
        default='wavenet_diffusion_logs/2023-09-21T23-47-07_mm_transformer_v4/checkpoints/epoch=000117.ckpt',
        help="load checkpoint",
    )

    parser.add_argument(
        "--base",
        type=str,
        required=False,
        default='wavenet_diffusion_logs/2023-09-21T23-47-07_mm_transformer_v4/configs/2023-09-23T23-52-13-project.yaml',
        help="yaml dir",
    )

    parser.add_argument(
        "--vqvae",
        type=str,
        required=False,
        default='/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_dance/weight/musicgen_vqvae.bin',
        help="load checkpoint of pretrained vqvae",
    )

    parser.add_argument(
        "--motion_vqvae_ckpt",
        type=str,
        required=False,
        default='/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_dance/weight/mm_vqvae.ckpt',
        help="load checkpoint of pretrained motion vqvae",
    )

    parser.add_argument(
        "--motion_vqvae_config",
        type=str,
        required=False,
        default='/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_motion_diffusion/configs/mm_vqvae_v6.yaml',
        help="load checkpoint of pretrained motion vqvae config",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        required=False,
        default=0.7,
        help="drop out probability",
    )

    args = parser.parse_args()

    save_path = args.save_path
    guidance_scale = args.guidance_scale
    music_dir = args.music_dir
    motion_dir = args.motion_dir
    duration = args.duration
    dropout_prob = args.dropout

    music_ignore_list = []
    music_data_list = []

    assert os.path.exists(args.meta_dir)

    with cs.open(pjoin(args.meta_dir, f'music4all_ignore.txt'), 'r') as f:
        for line in f.readlines():
            music_ignore_list.append(line.strip())

    with cs.open(pjoin(args.meta_dir, f'music4all_test.txt'), "r") as f:
        for line in f.readlines():
            if line.strip() in music_ignore_list:
                continue
            music_data_list.append(line.strip())

    print('number of testing data:', len(music_data_list))
    os.makedirs(save_path, exist_ok=True)

    pkg = torch.load(args.vqvae, map_location='cpu')
    cfg = OmegaConf.create(pkg['xp.cfg'])
    music_vqvae = get_compression_model(cfg)
    music_vqvae.load_state_dict(pkg['best_state'])
    music_vqvae.eval()

    motion_vqvae_configs = OmegaConf.load(args.motion_vqvae_config)
    motion_vqvae: pl.LightningModule = instantiate_from_config(motion_vqvae_configs.model)
    pl_sd = torch.load(args.motion_vqvae_ckpt, map_location='cpu')
    motion_vqvae.load_state_dict(pl_sd['state_dict'])
    motion_vqvae.eval()

    config = OmegaConf.load(args.base)
    model = load_model_from_config(config, args.ckpt)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    music_vqvae = music_vqvae.to(device)
    motion_vqvae = motion_vqvae.to(device)

    motion_mean = np.load(pjoin(motion_dir, 'Mean.npy'))
    motion_std = np.load(pjoin(motion_dir, 'Std.npy'))
    kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                       [9, 13, 16, 18, 20]]

    # Some text prompt
    text_prompt_list = ["piano", "rock", "pop", "drum", "techno", "guitar",
                        'Long piano music', 'Classical, orchestral music', 'Piano with other instruments', 'Soft jazz'
                        'Orchestra with violins', 'Soft piano', 'African Drums, Rhythm', 'Electro House']
    text_prompt_list = [f'this is a {s} music' for s in text_prompt_list]

    text_df = pd.read_csv(pjoin(args.meta_dir, 'text_prompt.csv'), index_col=0)
    for music_id in music_data_list:
        tag_list = text_df.loc[music_id, 'tags']
        if pd.isna(tag_list):
            tag_list = ['nan value']
            tag_is_empty = True
        else:
            tag_list = tag_list.split('\t')
            tag_is_empty = False

        song_name = text_df.loc[music_id, 'name']
        album_name = text_df.loc[music_id, 'album']
        key = text_df.loc[music_id, 'key']

        # choose for tempo descriptor
        tempo = text_df.loc[music_id, 'tempo']
        if tempo < 60:
            tempo_description = 'very slow'
        elif 60 <= tempo < 75:
            tempo_description = 'slow'
        elif 75 <= tempo < 110:
            tempo_description = 'moderate'
        elif 110 <= tempo < 150:
            tempo_description = 'fast'
        else:
            tempo_description = 'very fast'

        # choose for energy descriptor
        energy = text_df.loc[music_id, 'energy']
        if energy < 0.1:
            energy_description = 'very peaceful'
        elif 0.1 <= energy < 0.4:
            energy_description = 'peaceful'
        elif 0.4 <= energy < 0.7:
            energy_description = 'moderate'
        elif 0.7 <= energy < 0.95:
            energy_description = 'intense'
        else:
            energy_description = 'very intense'

        # drop out some tags
        p = random.uniform(0, 1)
        if p < dropout_prob:
            song_name = None
        p = random.uniform(0, 1)
        if p < dropout_prob:
            album_name = None
        p = random.uniform(0, 1)
        if p < dropout_prob:
            key = None
        p = random.uniform(0, 1)
        if p < dropout_prob:
            tempo_description = None
        p = random.uniform(0, 1)
        if p < dropout_prob:
            energy_description = None
        filtered_tag_list = []
        for tag in tag_list:
            p = random.uniform(0, 1)
            if p > dropout_prob:
                filtered_tag_list.append(tag)

        # construct phrases
        name_choices = [f'with the title of {song_name}', f'named {song_name}']

        verb_choices = ['selected']
        album_choices = [f'{random.choice(verb_choices)} from {album_name}']

        key_choices = [f'all set in the key of {key}']

        noun_choices = ['tempo']
        tempo_choices = [f'with a {tempo_description} {random.choice(noun_choices)}']

        noun_choices = ['intensity']
        energy_choices = [f'which is {energy_description}']

        noun_choices = ['genre']
        if len(filtered_tag_list) == 0:
            filtered_tag_list = [random.choice(tag_list)]  # ensure at least have 1 tag
        if len(filtered_tag_list) == 1:
            tag_string = filtered_tag_list[0]
        else:
            tag_string = ', '.join(filtered_tag_list[:-1]) + f' and {filtered_tag_list[-1]}'
        tag_choices = [f'the {random.choice(noun_choices)} of the music is {tag_string}']

        phrase_name = random.choice(name_choices)
        phrase_album = random.choice(album_choices)
        phrase_key = random.choice(key_choices)
        phrase_tempo = random.choice(tempo_choices)
        phrase_energy = random.choice(energy_choices)
        phrase_tag = random.choice(tag_choices)

        text_prompt = []
        if song_name is not None:
            text_prompt.append(phrase_name)
        if album_name is not None:
            text_prompt.append(phrase_album)
        if key is not None:
            text_prompt.append(phrase_key)
        if tempo_description is not None:
            text_prompt.append(phrase_tempo)
        if energy_description is not None:
            text_prompt.append(phrase_energy)
        if not tag_is_empty:
            text_prompt.append(phrase_tag)

        if len(text_prompt) == 0:
            print('zero length text prompt!')
            continue

        random.shuffle(text_prompt)
        text_prompt = ', '.join(text_prompt)

        text_prompt_list.append(text_prompt)

    count = 0
    total_num = len(text_prompt_list)
    while count < total_num:
        # text condition
        music_id = text_prompt_list[count]
        cond = [music_id]
        print('Conditioned on text: ', music_id)

        with torch.no_grad():
            batch = {
                'music_code': None,
                'motion_code': None,
                'text': cond
            }

            music_gen, motion_gen, _, _, _ = model.generate_sample(
                batch,
                duration=duration,
                conditional_guidance_scale=args.guidance_scale
            )

            waveform_gen = music_vqvae.decode(music_gen)
            waveform_gen = waveform_gen.cpu().squeeze()
            motion_gen = motion_vqvae.decode_from_code(music_gen, motion_gen)
            joint_gen = motion_vec_to_joint(motion_gen, motion_mean, motion_std)[0]

            os.makedirs(save_path, exist_ok=True)
            if ' ' in music_id:
                music_id = "_".join(music_id.split(' '))
            if '/' in music_id:
                music_id = "_".join(music_id.split('/'))

            music_filename = "music.mp3"
            music_path = os.path.join(save_path, music_filename)
            try:
                sf.write(music_path, waveform_gen, 32000)
            except:
                count += 1
                continue

            motion_filename = "motion.mp4"
            motion_path = pjoin(save_path, motion_filename)
            try:
                skel_animation.plot_3d_motion(
                    motion_path, kinematic_chain, joint_gen, title='None', vbeat=None,
                    fps=20, radius=4
                )
            except:
                count += 1
                continue

            video_filename = "%s.mp4" % music_id
            video_path = pjoin(save_path, video_filename)
            subprocess.call(
                f"ffmpeg -i {motion_path} -i {music_path} -c copy {video_path}",
                shell=True)

        count += 1
