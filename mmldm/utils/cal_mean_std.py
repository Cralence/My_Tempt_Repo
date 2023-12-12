import os
import torch
import codecs as cs
from os.path import join as pjoin


data_dir = "/nobackup/users/yiningh/yh/music4all/music4all_processed_feature"
meta_dir = "/nobackup/users/yiningh/yh/music4all"

# prepare for data
music_data = []
music_ignore = []

for split in ['train', 'test', 'val']:
    with cs.open(pjoin(meta_dir, f'music4all_{split}.txt'), "r") as f:
        for line in f.readlines():
            if line.strip() in music_ignore:
                continue
            if os.path.exists(pjoin(data_dir, line.strip() + '.pth')):
                music_data.append(line.strip())

print('total data num: ', len(music_data))

sum_feature = torch.zeros(128)
sum_squred = torch.zeros(128)

for i, music_id in enumerate(music_data):
    feature_path = pjoin(data_dir, music_id + '.pth')
    feature_dict = torch.load(feature_path)

    feature = feature_dict['music_token']  # (1, dim, n)
    feature = feature.squeeze()
    print('%d/%d' % (i+1, len(music_data)))

    sum_feature += torch.mean(feature, dim=-1)
    sum_squred += torch.mean(feature ** 2, dim=-1)


mean = sum_feature / len(music_data)
std = (sum_squred / len(music_data) - mean ** 2) ** 0.5

torch.save(mean, pjoin(meta_dir, 'Feature_mean.pth'))
torch.save(std, pjoin(meta_dir, 'Feature_std.pth'))

print('mean:')
print(mean)

print('std')
print(std)



