import pandas as pd
import os
import torch

meta_dir = '/nobackup/users/yiningh/yh/music4all'
feature_dir = '/nobackup/users/yiningh/yh/music4all/music4all_feature'

df = pd.read_csv(os.path.join(meta_dir, 'id_metadata.csv'), sep='\t')

for i in range(len(df)):
    music_id = df.loc[i, 'id']
    tempo = df.loc[i, 'tempo']
    if tempo > 150 and os.path.exists(os.path.join(feature_dir, music_id + '.pth')):
        feature_dict = torch.load(os.path.join(feature_dir, music_id + '.pth'))
        detected_bpm = len(feature_dict['beat']) * 2

        if detected_bpm < tempo and tempo - detected_bpm > 20:
            df.loc[i, 'tempo'] = detected_bpm
            print(f'{music_id}: change from {tempo} to {detected_bpm}')

df.to_csv(os.path.join(meta_dir, 'id_metadata_new.csv'), sep='\t', index=False)
