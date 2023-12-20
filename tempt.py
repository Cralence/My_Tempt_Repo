import torch

ckpt_1 = '../../final_version_2/My_Tempt_Repo/mm_transformer_logs/train_music_motion/checkpoints/epoch\=000025.ckpt'
ckpt_2 = '../../final_version_2/My_Tempt_Repo/mm_transformer_logs/2023-12-14T20-48-06_train_lm/checkpoints/last.ckpt'

a = torch.load(ckpt_1, map_location='cpu')['state_dict']
b = torch.load(ckpt_2, map_location='cpu')['state_dict']

for k in a.keys():
    print(f'{k}, {torch.sum(a[k] - b[k])}')

