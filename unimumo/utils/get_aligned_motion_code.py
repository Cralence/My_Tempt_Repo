import torch
import os
from os.path import join as pjoin
import codecs as cs
import numpy as np
import librosa
import random
from unimumo.alignment import visual_beat, interpolation
from unimumo.motion import motion_process
from unimumo.motion.motion_process import recover_from_ric
from dtw import *
from omegaconf import OmegaConf
from unimumo.util import instantiate_from_config
from pytorch_lightning import seed_everything
import argparse

'''
This is for using mm_vqvae_v6 to extract motion code that is aligned with music.
The vqvae uses encodec codebook, but encode motion separately
'''


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

music_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/spotify_music/audios'
music_metadata_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/spotify_music/'
motion_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_dance/motion_data'
feature_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/spotify_music/spotify_music_beat'
motion_feature_save_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_dance/motion_data/aligned_motion_code_spotify'

ckpt = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_dance/weight/mm_vqvae.ckpt'
yaml_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_motion_diffusion/configs/mm_vqvae_v6.yaml'

num_pair_each_music = 5

os.makedirs(motion_feature_save_dir, exist_ok=True)
seed_everything(2023)

config = OmegaConf.load(yaml_dir)
model = load_model_from_config(config, ckpt)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
model.eval()


ignore = []
motion_data = {'train': [], 'test': [], 'val': []}
aist = []
dancedb = []
humanml3d = []
motion_mean = np.load(pjoin(motion_dir, 'Mean.npy'))
motion_std = np.load(pjoin(motion_dir, 'Std.npy'))

with cs.open(pjoin(motion_dir, 'ignore_list.txt'), "r") as f:
    for line in f.readlines():
        ignore.append(line.strip())
for split in ['train', 'test', 'val']:

    with cs.open(pjoin(motion_dir, f'humanml3d_{split}.txt'), "r") as f:
        for line in f.readlines():
            if line.strip() in ignore:
                continue
            if not os.path.exists(pjoin(motion_dir, split, 'joint_vecs', line.strip() + '.npy')):
                continue
            motion_data[split].append(line.strip())
            humanml3d.append(line.strip())

    with cs.open(pjoin(motion_dir, f'aist_{split}.txt'), "r") as f:
        for line in f.readlines():
            if line.strip() in ignore:
                continue
            if not os.path.exists(pjoin(motion_dir, split, 'joint_vecs', line.strip() + '.npy')):
                continue
            for _ in range(30):
                motion_data[split].append(line.strip())
            aist.append(line.strip())
    with cs.open(pjoin(motion_dir, f'dancedb_{split}.txt'), "r") as f:
        for line in f.readlines():
            if line.strip() in ignore:
                continue
            if not os.path.exists(pjoin(motion_dir, split, 'joint_vecs', line.strip() + '.npy')):
                continue
            for _ in range(30):
                motion_data[split].append(line.strip())
            dancedb.append(line.strip())

music_data = []
for split in ['train', 'test', 'val']:
    with cs.open(pjoin(music_metadata_dir, f'spotify_{split}.txt'), "r") as f:
        for line in f.readlines():
            if not os.path.exists(pjoin(music_dir, line.strip() + '.mp3')):
                continue
            if os.path.exists(pjoin(feature_dir, line.strip() + '.pth')):
                music_data.append([line.strip(), split])

start_idx = int(args.start * len(music_data))
end_idx = int(args.end * len(music_data))
music_data = music_data[start_idx:end_idx]

print(f'total music: {len(music_data)}')
num_motion = len(motion_data['train']) + len(motion_data['val']) + len(motion_data['test'])
print(f'total motion: {num_motion}')

print(f'length of dance: {len(dancedb) + len(aist)}')
print(f'length of non dance: {len(humanml3d)}')

for data_idx, (music_id, split) in enumerate(music_data):
    # check whether the music has already been paired
    generated_motion_list = os.listdir(motion_feature_save_dir)
    generated_motion_list = [f for f in generated_motion_list if f[:len(music_id)] == music_id]
    if len(generated_motion_list) >= num_pair_each_music:
        print(f'{data_idx + 1}/{len(music_data)} already exists!')
        continue

    print(f'{data_idx + 1}/{len(music_data)}', end=' ')
    music_path = pjoin(music_dir, f'{music_id}.mp3')
    waveform, sr = librosa.load(music_path, sr=32000)
    music_duration = 30
    max_motion_length = int(music_duration * 20)
    max_music_length = int(music_duration * 32000)

    waveform = torch.FloatTensor(waveform)
    if waveform.shape[0] != max_music_length:
        if waveform.shape[0] > max_music_length:
            waveform = waveform[:max_music_length]
        else:
            zero_pad = torch.zeros(max_music_length)
            zero_pad[:waveform.shape[0]] = waveform
            waveform = zero_pad
    waveform = waveform[None, None, ...]
    print(f'music shape: {waveform.shape}', end=' ')

    feature_dict = torch.load(pjoin(feature_dir, f'{music_id}.pth'))

    for pair_idx in range(num_pair_each_music):
        while True:
            mbeat = feature_dict['beat']
            random_motion_idx = random.randint(0, len(motion_data[split]) - 1)
            motion_name = motion_data[split][random_motion_idx]

            motion = np.load(pjoin(motion_dir, split, 'joint_vecs', motion_name + '.npy'))
            if motion_name in humanml3d or motion_name in dancedb:
                motion_length = motion.shape[0]
                # if motion length longer than 10 sec
                aug = max_motion_length // motion_length
                if aug < 1:
                    start_idx = random.randint(0, motion_length - max_motion_length)
                    motion = motion[start_idx:start_idx+max_motion_length]
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
                    if max_motion_length - max_repeat*motion.shape[0] > 50:
                        max_repeat += 1
                    motion = np.tile(motion, (max_repeat, 1))
                    # length = motion.shape[0]
            else:
                motion_length = motion.shape[0] // 3 # 60 fps -> 20 fps
                if max_motion_length // motion_length < 1:
                    start_idx = random.randint(0, motion_length - max_motion_length)
                    motion = motion[start_idx*3:(start_idx+max_motion_length)*3:3]
                    # length = self.max_motion_length
                elif max_motion_length // motion_length == 1:
                    motion = motion[::3]
                    # length = motion.shape[0]
                else:
                    max_repeat = max_motion_length // motion_length + 1
                    motion = motion[::3]
                    # repeat = random.randint(1, max_repeat)
                    motion = np.tile(motion, (max_repeat, 1))

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
                print('bad motion')
                continue

            mbeats = np.zeros((max_motion_length))
            for beat in mbeat:
                if beat < len(mbeats):
                    mbeats[beat] = 1


            '''
            num_vbeat = np.sum(vbeats)
            num_mbeat = np.sum(mbeats)
            smaller = num_mbeat if num_mbeat < num_vbeat else num_vbeat
            larger = num_mbeat if num_mbeat > num_vbeat else num_vbeat
            if (larger - smaller) / larger > 0.2 and abs(larger - smaller*2) / larger > 0.2:
                print(f'Not align: num mbeat: {num_mbeat}, num vbeat: {num_vbeat}')
                continue
            '''

            try:
                alignment = dtw(
                    vbeats, mbeats, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "d"))
                wq = warp(alignment, index_reference=False)
                final_motion = interpolation.interp(motion, wq)
                break
            except ValueError:
                print('bad', motion.shape)
                continue

        motion = (final_motion - motion_mean) / motion_std
        motion = torch.FloatTensor(motion)  # T, D


        if motion.shape[0] != max_motion_length:  # pad the motion into the same shape
            if motion.shape[0] > max_motion_length:
                motion = motion[:max_motion_length, :]
            else:
                zero_pad = torch.zeros((max_motion_length, 263))
                zero_pad[:motion.shape[0], :] = motion
                motion = zero_pad

        motion = motion[None, ...]

        zero_waveform = torch.zeros_like(waveform)

        input_batch = {'motion': motion.to(device), 'music': zero_waveform.to(device)}

        music_emb, motion_emb = model.encode(input_batch)
        #
        # q_res_music = model.quantizer(music_emb, 50)  # 50 is the fixed sample rate
        # music_code = model.quantizer.encode(music_emb)
        #
        motion_code = model.quantizer.encode(motion_emb)
        #
        # motion_quantized_representation = model.quantizer.decode(motion_code)
        # music_quantized_representation = torch.zeros_like(motion_quantized_representation).to(device)
        # motion_recon = model.decode(music_quantized_representation, motion_quantized_representation).cpu()
        #
        # curr_loss = torch.nn.functional.mse_loss(motion, motion_recon)
        # print(f'split: {split}, {data_idx}/{len(music_data)}, loss: {curr_loss}')
        #
        motion_token = motion_code.squeeze()
        print(f'motion {pair_idx} feature shape: ', motion_token.shape, end='; ')
        motion_token = motion_token.cpu()

        motion_token_save_path = pjoin(motion_feature_save_dir, music_id + f'_!motion_code!_{motion_name}.pth')
        torch.save(motion_token, motion_token_save_path)  # 4, 1500
        '''
        if curr_loss > 0.3:
            joint = motion_vec_to_joint(motion_recon, motion_mean, motion_std)
            gt_joint = motion_vec_to_joint(motion, motion_mean, motion_std)
    
            os.makedirs(save_video_dir, exist_ok=True)
    
            motion_filename = f'bad_motion_{motion_name}_recon.mp4'
            motion_save_path = pjoin(save_video_dir, motion_filename)
            skel_animation.plot_3d_motion(
                motion_save_path, kinematic_chain, joint[0], title='None', vbeat=None,
                fps=20, radius=4
            )
    
            gt_motion_filename = f'bad_motion_{motion_name}_gt.mp4'
            motion_save_path = pjoin(save_video_dir, gt_motion_filename)
            skel_animation.plot_3d_motion(
                motion_save_path, kinematic_chain, gt_joint[0], title='None', vbeat=None,
                fps=20, radius=4
            )
        '''
    print('')



