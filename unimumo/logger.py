import os
import soundfile as sf
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from unimumo.motion import skel_animation
import subprocess
from os.path import join as pjoin
class MusicMotionLogger(Callback):
    def __init__(self, batch_frequency=2000, max_videos=4, increase_log_steps=True,
                 disabled=False, log_on_batch_idx=False, log_first_step=False,
                 unconditional_guidance_scale=1., motion_dir=None):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_videos = max_videos
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.log_first_step = log_first_step
        self.kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                               [9, 13, 16, 18, 20]]
        self.motion_dir = motion_dir
        self.mean = np.load(pjoin(self.motion_dir, 'Mean.npy'))
        self.std = np.load(pjoin(self.motion_dir, 'Std.npy'))

    @rank_zero_only
    def log_local(self, save_dir, split, music, motion, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "video_log", split)
        for i in range(music.shape[0]):
            music_filename = "gs-{:06}_e-{:06}_b-{:06}_{}.wav".format(global_step, current_epoch, batch_idx, i)
            music_path = os.path.join(root, music_filename)
            os.makedirs(os.path.split(music_path)[0], exist_ok=True)
            # print(music[i].shape)
            sf.write(music_path, music[i].reshape(-1, 1), 16000)

            motion_filename = "gs-{:06}_e-{:06}_b-{:06}_motion_{}.mp4".format(global_step, current_epoch, batch_idx, i)
            motion_path = os.path.join(root, motion_filename)
            os.makedirs(os.path.split(motion_path)[0], exist_ok=True)
            skel_animation.plot_3d_motion(motion_path, self.kinematic_chain, motion[i],
                                          title="None", vbeat=None, fps=20,
                                          radius=4)
            video_filename = "gs-{:06}_e-{:06}_b-{:06}_video_{}.mp4".format(global_step, current_epoch, batch_idx, i)
            video_path = os.path.join(root, video_filename)
            os.makedirs(os.path.split(video_path)[0], exist_ok=True)
            subprocess.call(
                f"ffmpeg -i {motion_path} -i {music_path} -c:v copy -c:a aac {video_path}",
                shell=True)

    def log_videos(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "generate_sample") and
                callable(pl_module.generate_sample) and
                self.max_videos > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                music, motion = pl_module.generate_sample([batch],
                      unconditional_guidance_scale=self.unconditional_guidance_scale,
                      motion_mean=self.mean,
                      motion_std=self.std)
            N = min(music.shape[0], self.max_videos)
            music = music[:N]
            motion = motion[:N]

            self.log_local(pl_module.logger.save_dir, split, music,motion,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_videos(pl_module, batch, batch_idx, split="train")