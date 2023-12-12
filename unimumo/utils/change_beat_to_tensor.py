import torch
import os

beat_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music4all/music4all_beat'

beat_list = os.listdir(beat_dir)
for i, beat_file in enumerate(beat_list):
    feature_path = os.path.join(beat_dir, beat_file)
    feature = torch.load(feature_path)
    beat = feature['beat']
    if isinstance(beat, torch.Tensor):
        continue
    beat = torch.LongTensor(beat)
    feature['beat'] = beat
    torch.save(feature, feature_path)
    print(f'{i + 1}/{len(beat_dir) + 1}, {beat[:10]}')
