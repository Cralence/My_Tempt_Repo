import os
import soundfile as sf
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from unimumo.motion import skel_animation
import subprocess
from os.path import join as pjoin
import random
from omegaconf import OmegaConf

from unimumo.audio.audiocraft.models.builders import get_compression_model


class MusicMotionTextLogger(Callback):
    def __init__(self, pretrained_vqvae_ckpt, unconditional_guidance_scale=2.5, duration=10, batch_frequency=2000, max_videos=4, increase_log_steps=True,
                 disabled=False, log_on_batch_idx=False, log_first_step=False,
                 motion_dir=None, max_audio_logged=60, log_epoch_cycle=21, motion_fps=20):
        super().__init__()
        self.pretrained_vqvae_ckpt = pretrained_vqvae_ckpt
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.duration = duration
        self.batch_freq = batch_frequency
        self.max_videos = max_videos
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                               [9, 13, 16, 18, 20]]
        self.motion_dir = motion_dir
        self.mean = np.load(pjoin(self.motion_dir, 'Mean.npy'))
        self.std = np.load(pjoin(self.motion_dir, 'Std.npy'))
        self.motion_fps = motion_fps

        self.max_audio_logged = max_audio_logged
        self.log_epoch_cycle = log_epoch_cycle
        self.log_cycle_count = -1

    @rank_zero_only
    def log_local(self, save_dir, split, music, motion,
                  gt_music, gt_motion, text, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "video_log", split)
        for i in range(music.shape[0]):
            music_filename = "e-{:06}_b-{:06}_music_{}_gen.mp3".format(current_epoch, batch_idx, i)
            music_path = os.path.join(root, music_filename)
            os.makedirs(os.path.split(music_path)[0], exist_ok=True)
            # print(music[i].shape)
            try:
                sf.write(music_path, music[i].squeeze().cpu().detach().numpy(), 32000)
            except:
                print(f'{music_path} cannot be saved')
                continue

            gt_music_filename = "e-{:06}_b-{:06}_music_{}_ref.mp3".format(current_epoch, batch_idx, i)
            gt_music_path = os.path.join(root, gt_music_filename)
            os.makedirs(os.path.split(gt_music_path)[0], exist_ok=True)
            # print(music[i].shape)
            try:
                sf.write(gt_music_path, gt_music[i].squeeze().cpu().detach().numpy(), 32000)
            except:
                print(f'{gt_music_path} cannot be saved')
                continue

            motion_filename = "e-{:06}_b-{:06}_motion_{}.mp4".format(current_epoch, batch_idx, i)
            motion_path = os.path.join(root, motion_filename)
            os.makedirs(os.path.split(motion_path)[0], exist_ok=True)
            skel_animation.plot_3d_motion(motion_path, self.kinematic_chain, motion[i],
                                          title="None", vbeat=None, fps=self.motion_fps,
                                          radius=4)
            video_filename = "e-{:06}_b-{:06}_video_{}.mp4".format(current_epoch, batch_idx, i)
            video_path = os.path.join(root, video_filename)
            os.makedirs(os.path.split(video_path)[0], exist_ok=True)
            subprocess.call(
                f"ffmpeg -i {motion_path} -i {music_path} -c copy {video_path}",
                shell=True)

            gt_motion_filename = "e-{:06}_b-{:06}_motion_{}_gt.mp4".format(current_epoch, batch_idx, i)
            gt_motion_path = os.path.join(root, gt_motion_filename)
            os.makedirs(os.path.split(gt_motion_path)[0], exist_ok=True)
            skel_animation.plot_3d_motion(gt_motion_path, self.kinematic_chain, gt_motion[i],
                                          title="None", vbeat=None, fps=self.motion_fps,
                                          radius=4)
            gt_video_filename = "e-{:06}_b-{:06}_video_{}_gt.mp4".format(current_epoch, batch_idx, i)
            gt_video_path = os.path.join(root, gt_video_filename)
            os.makedirs(os.path.split(gt_video_path)[0], exist_ok=True)
            subprocess.call(
                f"ffmpeg -i {gt_motion_path} -i {gt_music_path} -c copy {gt_video_path}",
                shell=True)

            os.system(f'rm {music_path}')

        # remvoe old videos
        audio_list = os.listdir(root)
        audio_list.sort()
        if len(audio_list) > self.max_audio_logged:
            to_remove = audio_list[:-self.max_audio_logged]
            for name in to_remove:
                remove_path = os.path.join(root, name)
                if os.path.exists(remove_path):
                    os.system('rm %s' % remove_path)
                    print('removed: %s' % remove_path)
                else:
                    print('not found: %s' % remove_path)

    def log_videos(self, pl_module, batch, batch_idx, split="train", random_pick=False):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                self.max_videos > 0):

            self.log_cycle_count += 1
            if not self.log_cycle_count % self.log_epoch_cycle == 0:
                print('In log cycle %d, do not log results' % self.log_cycle_count)
                return

            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                music_emb_gen, joint_gen, music_emb_ref, joint_ref, text = pl_module.generate_sample(
                    [batch],
                    motion_mean=self.mean,
                    motion_std=self.std,
                    unconditional_guidance_scale=self.unconditional_guidance_scale,
                    duration=self.duration)

            N = min(music_emb_gen.shape[0], self.max_videos)
            if random_pick:
                idx = torch.LongTensor(random.sample(range(music_emb_gen.shape[0]), N))
                music_emb_gen = music_emb_gen[idx]
                joint_gen = joint_gen[idx.numpy()]
                music_emb_ref = music_emb_ref[idx]
                joint_ref = joint_ref[idx.numpy()]
                tempt_list = []
                for num in idx:
                    tempt_list.append(text[num])
                text = tempt_list
            else:
                music_emb_gen = music_emb_gen[:N]
                joint_gen = joint_gen[:N]
                music_emb_ref = music_emb_ref[:N]
                joint_ref = joint_ref[:N]
                text = text[:N]

            # load VQVAE model
            pkg = torch.load(self.pretrained_vqvae_ckpt, map_location='cpu')
            cfg = OmegaConf.create(pkg['xp.cfg'])
            model = get_compression_model(cfg)
            model.load_state_dict(pkg['best_state'])
            model.eval().to(music_emb_gen.device)

            with torch.no_grad():
                music_gen = model.decoder(music_emb_gen)
                music_ref = model.decoder(music_emb_ref)

            self.log_local(pl_module.logger.save_dir, split, music_gen, joint_gen, music_ref, joint_ref, text,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_videos(pl_module, batch, batch_idx, split="train")

        if batch_idx == 0:
            model_size = torch.cuda.max_memory_allocated(device=None)
            for _ in range(30):
                model_size /= 2
            print('############### GPU memory used %.1f GB #################' % model_size)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_videos(pl_module, batch, batch_idx, split="val", random_pick=True)
