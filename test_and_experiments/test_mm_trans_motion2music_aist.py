from mmldm.util import instantiate_from_config
from omegaconf import OmegaConf
import argparse
import os
import torch
from os.path import join as pjoin
import soundfile as sf
import librosa
import numpy as np
from mmldm.audio.audiocraft.models.builders import get_compression_model
from mmldm.motion.motion_process import motion_vec_to_joint
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=False,
        help="The path to save model output",
        default="./test_motion2music",
    )

    parser.add_argument(
        "--aist_dir",
        type=str,
        required=False,
        help="The path to music data dir",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/aist_plusplus_final"
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
        default=3.0,
        help="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
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
        default=None,
        help="load checkpoint",
    )

    parser.add_argument(
        "--base",
        type=str,
        required=False,
        default=None,
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

    args = parser.parse_args()

    save_path = args.save_path
    music_save_path = pjoin(save_path, 'music')
    motion_save_path = pjoin(save_path, 'motion')
    video_save_path = pjoin(save_path, 'video')
    guidance_scale = args.guidance_scale
    aist_dir = args.aist_dir
    motion_dir = args.motion_dir

    # read all aist motion data
    motion_data = {}
    for split in ['train', 'val', 'test']:
        data_list = os.listdir(pjoin(motion_dir, split, 'joint_vecs'))
        data_list = [s.split('.')[0] for s in data_list]
        data_list = [s for s in data_list if s[0] == 'g']
        motion_data[split] = data_list
        print(data_list[:20])

    motion_mean = np.load(pjoin(motion_dir, 'Mean.npy'))
    motion_std = np.load(pjoin(motion_dir, 'Std.npy'))
    kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                       [9, 13, 16, 18, 20]]

    motion_id_list = []
    with open(pjoin(aist_dir, 'aist_motion_test_segment.txt'), 'r') as f:
        for line in f.readlines():
            motion_id_list.append(line.strip())

    print('number of testing data:', len(motion_id_list))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(music_save_path, exist_ok=True)
    os.makedirs(motion_save_path, exist_ok=True)
    os.makedirs(video_save_path, exist_ok=True)

    pkg = torch.load(args.vqvae, map_location='cpu')
    cfg = OmegaConf.create(pkg['xp.cfg'])
    music_vqvae = get_compression_model(cfg)
    music_vqvae.load_state_dict(pkg['best_state'])
    music_vqvae.eval()

    motion_vqvae_configs = OmegaConf.load(args.motion_vqvae_config)
    motion_vqvae = instantiate_from_config(motion_vqvae_configs.model)
    pl_sd = torch.load(args.motion_vqvae_ckpt, map_location='cpu')
    motion_vqvae.load_state_dict(pl_sd['state_dict'])
    motion_vqvae.eval()

    config = OmegaConf.load(args.base)
    model = load_model_from_config(config, args.ckpt)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    music_vqvae = music_vqvae.to(device)
    motion_vqvae = motion_vqvae.to(device)

    total_num = len(motion_id_list)
    print(f'total number of test data: {total_num}')
    count = 0
    print('train: ', len(motion_data['train']), 'test: ', len(motion_data['test']), 'val: ', len(motion_data['val']))
    while count < total_num:
        motion_id = motion_id_list[count]
        motion_id = motion_id.split('/')[-1].split('.')[0]
        print(f'{motion_id} -> ', end='')
        motion_name = '_'.join(motion_id.split('_')[:-1])
        motion_name = '_'.join(motion_name.split('_')[:2]) + '_cAll_' + '_'.join(motion_name.split('_')[-3:])
        seg_num = int(motion_id.split('_')[-1][3:])
        print(f'{motion_name}, segment {seg_num}, ', end='')

        if motion_name in motion_data['train']:
            motion_path = pjoin(motion_dir, 'train', 'joint_vecs', motion_name + '.npy')
            print('train')
        elif motion_name in motion_data['val']:
            motion_path = pjoin(motion_dir, 'val', 'joint_vecs', motion_name + '.npy')
            print('val')
        elif motion_name in motion_data['test']:
            motion_path = pjoin(motion_dir, 'test', 'joint_vecs', motion_name + '.npy')
            print('test')
        else:
            motion_path = None
        assert os.path.exists(motion_path)
        motion = np.load(motion_path)
        
        # motion is in aist, so downsample by 3
        motion = motion[::3]

        # go to the specific segment
        motion = motion[(seg_num - 1) * 40: seg_num * 40]
        
        motion = (motion - motion_mean) / motion_std
        motion = torch.tensor(motion)
        duration = 2
        empty_waveform = torch.zeros((1, 1, duration * 32000))
        
        motion = motion[None, ...]
        print(f'motion shape: {motion.shape}, empty music shape: {empty_waveform.shape}, ', end='')

        with torch.no_grad():
            input_batch = {'motion': motion.to(device), 'music': empty_waveform.to(device)}
            _, motion_emb = motion_vqvae.encode(input_batch)
            motion_code = motion_vqvae.quantizer.encode(motion_emb)
            print(f'motion code shape: {motion_code.shape}')

            music_gen = model.generate_single_modality(
                music_code=None,
                motion_code=motion_code,
                conditional_guidance_scale=args.guidance_scale
            )

            waveform_gen = music_vqvae.decode(music_gen).cpu().squeeze().numpy()
            joint_gen = motion_vec_to_joint(motion, motion_mean, motion_std)[0]

        os.makedirs(save_path, exist_ok=True)

        music_filename = "%s.mp3" % motion_id
        music_path = os.path.join(music_save_path, music_filename)
        try:
            sf.write(music_path, waveform_gen, 32000)
        except:
            print(f'{music_filename} cannot be saved.')
            count += 1
            continue

        motion_filename = "%s.mp4" % motion_id
        motion_path = pjoin(motion_save_path, motion_filename)
        try:
            skel_animation.plot_3d_motion(
                motion_path, kinematic_chain, joint_gen, title='None', vbeat=None,
                fps=20, radius=4
            )
        except:
            print(f'{motion_filename} cannot be saved.')
            count += 1
            continue

        video_filename = "%s.mp4" % motion_id
        video_path = pjoin(video_save_path, video_filename)
        try:
            subprocess.call(
                f"ffmpeg -i {motion_path} -i {music_path} -c copy {video_path}",
                shell=True)
        except:
            print(f'{video_path} cannot be saved.')
            count += 1
            continue

        count += 1
