import torch
import numpy as np
import codecs as cs
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from os.path import join as pjoin
import random
import pickle
class MotionDataset(Dataset):
    def __init__(self, split,
        motion_dir, max_motion_length=200):
        self.njoints = 22
        self.motion_dir = motion_dir
        self.split = split
        self.max_motion_length = max_motion_length
        self.humanml3d = []
        self.aist = []
        self.dancedb = []
        self.data = []
        self.ignore = []
        with cs.open(pjoin(self.motion_dir, 'ignore_list.txt'), "r") as f:
            for line in f.readlines():
                self.ignore.append(line.strip())
        with cs.open(pjoin(self.motion_dir, f'humanml3d_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                self.data.append(line.strip())
                self.humanml3d.append(line.strip())
        with cs.open(pjoin(self.motion_dir, f'aist_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                if line.strip() in self.ignore:
                    continue
                self.data.append(line.strip())
                self.aist.append(line.strip())
        with cs.open(pjoin(self.motion_dir, f'dancedb_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                self.data.append(line.strip())
                self.dancedb.append(line.strip())

        self.mean = np.load(pjoin(self.motion_dir, 'Mean.npy'))
        self.std = np.load(pjoin(self.motion_dir, 'Std.npy'))
        with open(pjoin(self.motion_dir, f'{self.split}_length.pickle'), 'rb') as f:
            self.length = pickle.load(f)
        print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.data[idx]
        motion = np.load(pjoin(self.motion_dir, self.split, 'joint_vecs', name + '.npy'))
        if name in self.humanml3d or name in self.dancedb:
            motion_length = self.length[name]
            # if motion length longer than 10 sec
            if self.max_motion_length // motion_length < 1:
                start_idx = random.randint(0, motion_length - self.max_motion_length)
                motion = motion[start_idx:start_idx+self.max_motion_length]
                length = self.max_motion_length

            elif self.max_motion_length // motion_length == 1:
                motion = motion
                length = motion_length

            else:
                max_repeat = self.max_motion_length // motion_length
                repeat = random.randint(1, max_repeat)
                motion = np.tile(motion, (repeat, 1))
                length = motion.shape[0]
        else:
            motion_length = self.length[name] // 3 # 60 fps -> 20 fps
            if self.max_motion_length // motion_length < 1:
                start_idx = random.randint(0, motion_length - self.max_motion_length)
                motion = motion[start_idx*3:(start_idx+self.max_motion_length)*3:3]
                length = self.max_motion_length

            elif self.max_motion_length // motion_length == 1:
                motion = motion[::3]
                length = motion.shape[0]

            else:
                max_repeat = self.max_motion_length // motion_length
                motion = motion[::3]
                repeat = random.randint(1, max_repeat)
                motion = np.tile(motion, (repeat, 1))
                length = motion.shape[0]

        motion = (motion - self.mean) / self.std
        # print(name, motion.shape, motion_length, length)
        assert motion.shape[0] == length
        return {'motion': motion, 'length': length}

if __name__ == "__main__":
    dataset = MotionDataset('train', '/data/motion_data')
    size = len(dataset)
    for i in range(size):
        data_dict = dataset.__getitem__(i)