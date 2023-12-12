import os

import random
import soundfile as sf
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from omegaconf import OmegaConf
import numpy as np
from os.path import join as pjoin
from mmldm.motion import skel_animation
import subprocess

from mmldm.audio.audiocraft.models.builders import get_compression_model
from mmldm.util import instantiate_from_config
from mmldm.motion.motion_process import recover_from_ric


class MusicLogger(Callback):
    def __init__(self, music_vqvae_path, motion_vqvae_config, motion_vqvae_path=None,
                 motion_dir=None, motion_fps=20,
                 batch_frequency=2000, max_audio=4, increase_log_steps=True,
                 disabled=False, log_on_batch_idx=False, log_first_step=False,
                 duration=10, sr=32000, max_audio_logged=60, log_epoch_cycle=21,
                 conditional_guidance_scale=None):
        super().__init__()
        self.music_vqvae_path = music_vqvae_path
        self.motion_vqvae_path = motion_vqvae_path
        self.motion_vqvae_config = OmegaConf.load(motion_vqvae_config)
        self.conditional_guidance_scale = conditional_guidance_scale
        self.batch_freq = batch_frequency
        self.max_audios = max_audio
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.duration = duration
        self.sr = sr
        self.max_audio_logged = max_audio_logged
        self.log_epoch_cycle = log_epoch_cycle
        self.log_cycle_count = -1

        self.kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                                [9, 13, 16, 18, 20]]
        self.motion_dir = motion_dir
        self.mean = np.load(pjoin(self.motion_dir, 'Mean.npy'))
        self.std = np.load(pjoin(self.motion_dir, 'Std.npy'))
        self.motion_fps = motion_fps

    def motion_vec_to_joint(self, vec):
        # vec: [bs, 200, 263]
        mean = torch.tensor(self.mean).to(vec)
        std = torch.tensor(self.std).to(vec)
        vec = vec * std + mean
        joint = recover_from_ric(vec, joints_num=22)
        joint = joint.cpu().detach().numpy()
        return joint

    @rank_zero_only
    def log_local(self, save_dir, split, music, motion,
                  gt_music, gt_motion, current_epoch, batch_idx):
        root = os.path.join(save_dir, "video_log", split)
        print('save result root: ', root)
        for i in range(music.shape[0]):
            music_filename = "e-{:06}_b-{:06}_music_{}_gen.mp3".format(current_epoch, batch_idx, i)
            music_path = os.path.join(root, music_filename)
            os.makedirs(os.path.split(music_path)[0], exist_ok=True)
            try:
                sf.write(music_path, music[i].squeeze().cpu().detach().numpy(), 32000)
            except:
                print(f'{music_path} cannot be saved')
                continue

            if motion is None or gt_music is None or gt_motion is None:
                continue

            gt_music_filename = "e-{:06}_b-{:06}_music_{}_ref.mp3".format(current_epoch, batch_idx, i)
            gt_music_path = os.path.join(root, gt_music_filename)
            os.makedirs(os.path.split(gt_music_path)[0], exist_ok=True)
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

        # remove old audios
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

    @rank_zero_only
    def log_video_with_caption(self, save_dir, split,
                               gt_music, gt_motion, text_prompt, current_epoch, batch_idx):
        root = os.path.join(save_dir, "video_log", split)
        print('save result root: ', root)
        for i in range(gt_music.shape[0]):
            gt_music_filename = "music.mp3"
            gt_music_path = os.path.join(root, gt_music_filename)
            os.makedirs(os.path.split(gt_music_path)[0], exist_ok=True)
            try:
                sf.write(gt_music_path, gt_music[i].squeeze().cpu().detach().numpy(), 32000)
            except:
                print(f'{gt_music_path} cannot be saved')
                continue

            gt_motion_filename = "motion.mp4"
            gt_motion_path = os.path.join(root, gt_motion_filename)
            os.makedirs(os.path.split(gt_motion_path)[0], exist_ok=True)
            skel_animation.plot_3d_motion(gt_motion_path, self.kinematic_chain, gt_motion[i],
                                          title="None", vbeat=None, fps=self.motion_fps,
                                          radius=4)

            text = text_prompt[i]
            text = ' '.join(text.split('.'))
            text = '_'.join(text.split(' '))

            gt_video_filename = "e-{:06}_b-{:06}_video_{}_{}.mp4".format(current_epoch, batch_idx, i, text)
            gt_video_path = os.path.join(root, gt_video_filename)
            os.makedirs(os.path.split(gt_video_path)[0], exist_ok=True)
            subprocess.call(
                f"ffmpeg -i {gt_motion_path} -i {gt_music_path} -c copy {gt_video_path}",
                shell=True)

    def log_video(self, pl_module, batch, batch_idx, split="train", random_pick=False):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "generate_sample") and
                callable(pl_module.generate_sample) and
                self.max_audios > 0):
            self.log_cycle_count += 1
            if not self.log_cycle_count % self.log_epoch_cycle == 0:
                print('In log cycle %d, do not log results' % self.log_cycle_count)
                return

            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            if pl_module.stage == 'train_music_motion' or pl_module.stage == 'train_caption':
                with torch.no_grad():
                    music_gen, motion_gen, music_ref, motion_ref, text_prompt = pl_module.generate_sample(
                        batch,
                        duration=self.duration,
                        conditional_guidance_scale=self.conditional_guidance_scale
                    )

                N = min(music_gen.shape[0], self.max_audios)
                if random_pick:
                    idx = torch.LongTensor(random.sample(range(music_gen.shape[0]), N))
                    music_gen = music_gen[idx]
                    motion_gen = motion_gen[idx]
                    music_ref = music_ref[idx]
                    motion_ref = motion_ref[idx]
                    tempt_ls = []
                    for num in idx:
                        tempt_ls.append(text_prompt[num])
                else:
                    music_gen = music_gen[:N]
                    motion_gen = motion_gen[:N]
                    music_ref = music_ref[:N]
                    motion_ref = motion_ref[:N]

                # load VQVAE models
                pkg = torch.load(self.music_vqvae_path, map_location='cpu')
                cfg = OmegaConf.create(pkg['xp.cfg'])
                music_vqvae = get_compression_model(cfg)
                music_vqvae.load_state_dict(pkg['best_state'])

                motion_vqvae = instantiate_from_config(self.motion_vqvae_config.model)
                if self.motion_vqvae_path is not None:
                    pl_sd = torch.load(self.motion_vqvae_path, map_location='cpu')
                    motion_vqvae.load_state_dict(pl_sd['state_dict'])

                music_vqvae.eval()
                motion_vqvae.eval()
                with torch.no_grad():
                    music_vqvae.to(pl_module.device)
                    motion_vqvae.to(pl_module.device)
                    generated_music = music_vqvae.decode(music_gen)
                    ref_music = music_vqvae.decode(music_ref)
                    generated_motion = motion_vqvae.decode_from_code(music_gen, motion_gen)
                    ref_motion = motion_vqvae.decode_from_code(music_ref, motion_ref)
                generated_joint = self.motion_vec_to_joint(generated_motion)
                ref_joint = self.motion_vec_to_joint(ref_motion)

                self.log_local(pl_module.logger.save_dir, split,
                               generated_music, generated_joint, ref_music, ref_joint,
                               pl_module.current_epoch, batch_idx)

            if pl_module.stage == 'train_caption':
                with torch.no_grad():
                    text_prompt, music_ref, motion_ref = pl_module.generate_captions(batch)

                N = min(music_ref.shape[0], self.max_audios)
                if random_pick:
                    idx = torch.LongTensor(random.sample(range(music_ref.shape[0]), N))
                    music_ref = music_ref[idx]
                    motion_ref = motion_ref[idx]
                    tempt_ls = []
                    for num in idx:
                        tempt_ls.append(text_prompt[num])
                    text_prompt = tempt_ls
                else:
                    music_ref = music_ref[:N]
                    motion_ref = motion_ref[:N]
                    text_prompt = text_prompt[:N]

                # load VQVAE model
                pkg = torch.load(self.music_vqvae_path, map_location='cpu')
                cfg = OmegaConf.create(pkg['xp.cfg'])
                music_vqvae = get_compression_model(cfg)
                music_vqvae.load_state_dict(pkg['best_state'])

                motion_vqvae = instantiate_from_config(self.motion_vqvae_config.model)
                if self.motion_vqvae_path is not None:
                    pl_sd = torch.load(self.motion_vqvae_path, map_location='cpu')
                    motion_vqvae.load_state_dict(pl_sd['state_dict'])

                music_vqvae.eval()
                motion_vqvae.eval()
                with torch.no_grad():
                    music_vqvae.to(pl_module.device)
                    motion_vqvae.to(pl_module.device)

                    ref_music = music_vqvae.decode(music_ref)
                    ref_motion = motion_vqvae.decode_from_code(music_ref, motion_ref)
                    ref_joint = self.motion_vec_to_joint(ref_motion)

                self.log_video_with_caption(
                    pl_module.logger.save_dir, split, ref_music, ref_joint, text_prompt,
                    pl_module.current_epoch, batch_idx
                )

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_video(pl_module, batch, batch_idx, split="train")

        if batch_idx == 0:
            model_size = torch.cuda.max_memory_allocated(device=None)
            for _ in range(30):
                model_size /= 2
            print('############### GPU memory used %.1f GB #################' % model_size)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.disabled:
            self.log_video(pl_module, batch, batch_idx, split="val", random_pick=True)

