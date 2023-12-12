from omegaconf import OmegaConf
import torch
from mmldm.util import instantiate_from_config
from pytorch_lightning import seed_everything
import numpy as np
from os.path import join as pjoin
import os
import codecs as cs
import random
import librosa



config_path = '/nobackup/users/yiningh/yh/music_dance/music_motion_diffusion/configs/mm_vqvae_v6.yaml'
config_all = OmegaConf.load(config_path)
model_config = config_all['model']
ckpt_path = '/nobackup/users/yiningh/yh/music_dance/submission/music_motion_vae_logs/2023-08-11T12-14-09_mm_vqvae_v6/checkpoints/epoch=000943.ckpt'
motion_dir = "/nobackup/users/yiningh/yh/music_dance/motion_data"
music_dir = "/nobackup/users/yiningh/yh/music4all/audios"


model = instantiate_from_config(model_config)
sd = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(sd['state_dict'])
model.eval().cuda()

seed_everything(2023)

mean = np.load(pjoin(motion_dir, 'Mean.npy'))
std = np.load(pjoin(motion_dir, 'Std.npy'))
kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                   [9, 13, 16, 18, 20]]

motion_ignore_list = []
motion_data_list = []
aist = []
dancedb = []

with cs.open(pjoin(motion_dir, 'ignore_list.txt'), "r") as f:
    for line in f.readlines():
        motion_ignore_list.append(line.strip())
for split in ['train', 'test', 'val']:
    with cs.open(pjoin(motion_dir, f'aist_{split}.txt'), "r") as f:
        for line in f.readlines():
            if line.strip() in motion_ignore_list:
                continue
            if not os.path.exists(pjoin(motion_dir, split, 'joint_vecs', line.strip() + '.npy')):
                continue
            motion_data_list.append(line.strip() + '\t' + split)
            aist.append(line.strip() + '\t' + split)
    with cs.open(pjoin(motion_dir, f'dancedb_{split}.txt'), "r") as f:
        for line in f.readlines():
            if not os.path.exists(pjoin(motion_dir, split, 'joint_vecs', line.strip() + '.npy')):
                continue
            motion_data_list.append(line.strip() + '\t' + split)
            dancedb.append(line.strip() + '\t' + split)

print('number of testing data:', len(motion_data_list))

music_data_list = os.listdir(music_dir)
mean_ts = torch.load(pjoin(motion_dir, 'Feature_mean.pth'))
std_ts = torch.load(pjoin(motion_dir, 'Feature_std.pth'))
mean_ts = mean_ts[None, ..., None]
std_ts = std_ts[None, ..., None]

sum_feature = torch.zeros(128)
sum_squred = torch.zeros(128)
total_n_tokens = 0
for idx, motion_name in enumerate(motion_data_list):
    motion_name, split = motion_name.split('\t')
    motion = np.load(pjoin(motion_dir, split, 'joint_vecs', motion_name + '.npy'))

    if motion_name in aist:
        motion = motion[::3]

    motion_length = motion.shape[0]
    segment_length = 200
    num_segment = motion_length // segment_length + 1

    music_name = random.choice(music_data_list)
    waveform, _ = librosa.load(pjoin(music_dir, music_name), sr=32000)

    motion = (motion - mean) / std
    motion = torch.tensor(motion)

    padded_motion = torch.zeros((num_segment * segment_length, 263))
    padded_motion[:motion_length] = motion
    motion = padded_motion[None, ...]

    music_target_length = int(num_segment * 10 * 32000)
    waveform = torch.FloatTensor(waveform)
    print(f'music shape before processed: {waveform.shape}', end=', ')
    if waveform.shape[0] < music_target_length:
        num_full_repeat = music_target_length // waveform.shape[0]
        padded_waveform = torch.zeros(music_target_length)

        for i in range(num_full_repeat):
            padded_waveform[waveform.shape[0] * i:waveform.shape[0] * (i + 1)] = waveform

        padded_waveform[waveform.shape[0] * num_full_repeat:] = waveform[:music_target_length - waveform.shape[
            0] * num_full_repeat]
        waveform = padded_waveform
    else:
        waveform = waveform[:music_target_length]
    waveform = waveform[None, None, ...]


    with torch.no_grad():
        motion = motion.cuda()
        waveform = waveform.cuda()
        input_batch = {'motion': motion, 'music': waveform}

        music_emb, motion_emb = model.encode(input_batch)

        q_res_music = model.quantizer(music_emb, 50)  # 50 is the fixed sample rate
        q_res_motion = model.quantizer(motion_emb, 50)

        motion_recon = model.decode(q_res_music.x, q_res_motion.x)

        curr_loss = torch.nn.functional.mse_loss(motion, motion_recon)

        motion_token = q_res_motion.x.squeeze().cpu()
        motion_token_length = int(motion_length / 20 * 50)
        motion_token = motion_token[:, :motion_token_length]
        sum_feature += torch.sum(motion_token, dim=-1)
        sum_squred += torch.sum(motion_token ** 2, dim=-1)
        total_n_tokens += motion_token.shape[-1]

        normalized_token = (motion_token - mean_ts) / std_ts
        print(f'token shape: {motion_token.shape}', end=', ')
        print(f'before: {motion_token.mean()}, {motion_token.std()}', end=', ')
        print(f'after: mean: { normalized_token.mean()}, std: {normalized_token.std()}')



mean = sum_feature / total_n_tokens
std = (sum_squred / total_n_tokens - mean ** 2) ** 0.5

torch.save(mean, pjoin(motion_dir, 'Feature_mean.pth'))
torch.save(std, pjoin(motion_dir, 'Feature_std.pth'))

print('mean:')
print(mean)

print('std')
print(std)

