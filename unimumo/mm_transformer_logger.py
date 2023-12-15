import os
import sys
from os.path import join as pjoin
import random
import typing as tp
from typing import Any, Optional
import soundfile as sf
import torch
from omegaconf import OmegaConf
import numpy as np
import subprocess

from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning import LightningModule, Trainer

from unimumo.motion import skel_animation
from unimumo.audio.audiocraft.models.builders import get_compression_model
from unimumo.util import instantiate_from_config
from unimumo.motion.motion_process import recover_from_ric


class MusicLogger(Callback):
    def __init__(
        self, music_vqvae_path: str, motion_vqvae_config_path: str, motion_vqvae_path: str, motion_dir: str,
        motion_fps: int = 20, epoch_frequency: int = 10, batch_frequency: int = 2000,
        max_video_per_generation: int = 4, duration: float = 10., sr: int = 32000, max_video_logged: int = 60,
        conditional_guidance_scale: tp.Optional[float] = None, disabled: bool = False
    ):
        super().__init__()
        self.disabled = disabled

        # about music and motion vqvae
        self.music_vqvae_path = music_vqvae_path
        self.motion_vqvae_path = motion_vqvae_path
        self.motion_vqvae_config = OmegaConf.load(motion_vqvae_config_path)

        # about generation settings
        self.conditional_guidance_scale = conditional_guidance_scale
        self.duration = duration

        # about saving audios and videos
        self.sr = sr
        self.kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                                [9, 13, 16, 18, 20]]
        self.motion_dir = motion_dir
        self.mean = np.load(pjoin(self.motion_dir, 'Mean.npy'))
        self.std = np.load(pjoin(self.motion_dir, 'Std.npy'))
        self.motion_fps = motion_fps

        # about logging frequency and logging number
        self.epoch_freq = epoch_frequency
        self.batch_freq = batch_frequency
        self.max_videos_per_generation = max_video_per_generation
        self.max_videos_logged = max_video_logged

    def motion_vec_to_joint(self, vec: torch.Tensor) -> np.ndarray:
        # vec: [bs, 200, 263]
        mean = torch.tensor(self.mean).to(vec)
        std = torch.tensor(self.std).to(vec)
        vec = vec * std + mean
        joint = recover_from_ric(vec, joints_num=22)
        joint = joint.cpu().detach().numpy()
        return joint

    @rank_zero_only
    def log_local(
        self, save_dir: str, split: str, music: torch.Tensor, motion: np.ndarray,
        gt_music: torch.Tensor, gt_motion: np.ndarray, current_epoch: int, batch_idx: int
    ) -> None:
        root = os.path.join(save_dir, "video_log", split)
        print('save result root: ', root)

        for i in range(music.shape[0]):
            music_filename = "e-{:06}_b-{:06}_music_{}_gen.mp3".format(current_epoch, batch_idx, i)
            music_path = os.path.join(root, music_filename)
            os.makedirs(os.path.split(music_path)[0], exist_ok=True)
            try:
                sf.write(music_path, music[i].squeeze().cpu().detach().numpy(), self.sr)
            except:
                print(f'{music_path} cannot be saved')
                continue

            if motion is None or gt_music is None or gt_motion is None:
                continue

            gt_music_filename = "e-{:06}_b-{:06}_music_{}_ref.mp3".format(current_epoch, batch_idx, i)
            gt_music_path = os.path.join(root, gt_music_filename)
            os.makedirs(os.path.split(gt_music_path)[0], exist_ok=True)
            try:
                sf.write(gt_music_path, gt_music[i].squeeze().cpu().detach().numpy(), self.sr)
            except:
                print(f'{gt_music_path} cannot be saved')
                continue

            motion_filename = "e-{:06}_b-{:06}_motion_{}.mp4".format(current_epoch, batch_idx, i)
            motion_path = os.path.join(root, motion_filename)
            os.makedirs(os.path.split(motion_path)[0], exist_ok=True)
            skel_animation.plot_3d_motion(
                motion_path, self.kinematic_chain, motion[i], title="None", vbeat=None, fps=self.motion_fps, radius=4
            )

            video_filename = "e-{:06}_b-{:06}_video_{}.mp4".format(current_epoch, batch_idx, i)
            video_path = os.path.join(root, video_filename)
            os.makedirs(os.path.split(video_path)[0], exist_ok=True)
            subprocess.call(
                f"ffmpeg -i {motion_path} -i {music_path} -c copy {video_path}", shell=True
            )

            gt_motion_filename = "e-{:06}_b-{:06}_motion_{}_gt.mp4".format(current_epoch, batch_idx, i)
            gt_motion_path = os.path.join(root, gt_motion_filename)
            os.makedirs(os.path.split(gt_motion_path)[0], exist_ok=True)
            skel_animation.plot_3d_motion(
                gt_motion_path, self.kinematic_chain, gt_motion[i], title="None", vbeat=None, fps=self.motion_fps, radius=4
            )

            gt_video_filename = "e-{:06}_b-{:06}_video_{}_gt.mp4".format(current_epoch, batch_idx, i)
            gt_video_path = os.path.join(root, gt_video_filename)
            os.makedirs(os.path.split(gt_video_path)[0], exist_ok=True)
            subprocess.call(
                f"ffmpeg -i {gt_motion_path} -i {gt_music_path} -c copy {gt_video_path}", shell=True
            )

            # only keeps video
            os.system(f'rm {music_path}')
            os.system(f'rm {gt_music_path}')
            os.system(f'rm {motion_path}')
            os.system(f'rm {gt_motion_path}')

        # remove old videos
        video_list = os.listdir(root)
        video_list.sort()
        if len(video_list) > self.max_videos_logged:
            to_remove = video_list[:-self.max_videos_logged]
            for name in to_remove:
                remove_path = os.path.join(root, name)
                if os.path.exists(remove_path):
                    os.system(f'rm {remove_path}')
                    print('removed: %s' % remove_path)
                else:
                    print('not found: %s' % remove_path)

    @rank_zero_only
    def log_video_with_caption(
        self, save_dir: str, split: str, gt_music: torch.Tensor, gt_motion: np.ndarray,
        text_prompt: tp.List[str], current_epoch: int, batch_idx: int
    ) -> None:
        root = os.path.join(save_dir, "video_log", split)
        print('save result root: ', root)

        for i in range(gt_music.shape[0]):
            gt_music_filename = "music.mp3"
            gt_music_path = os.path.join(root, gt_music_filename)
            os.makedirs(os.path.split(gt_music_path)[0], exist_ok=True)
            try:
                sf.write(gt_music_path, gt_music[i].squeeze().cpu().detach().numpy(), self.sr)
            except:
                print(f'{gt_music_path} cannot be saved')
                continue

            gt_motion_filename = "motion.mp4"
            gt_motion_path = os.path.join(root, gt_motion_filename)
            os.makedirs(os.path.split(gt_motion_path)[0], exist_ok=True)
            skel_animation.plot_3d_motion(
                gt_motion_path, self.kinematic_chain, gt_motion[i], title="None", vbeat=None, fps=self.motion_fps, radius=4
            )

            # replace space and underscore characters
            text = text_prompt[i]
            text = ' '.join(text.split('.'))
            text = '_'.join(text.split(' '))
            # cut the text if it is too long, otherwise it cannot be used as filename
            text = text[:300]

            gt_video_filename = "e-{:06}_b-{:06}_video_{}_{}.mp4".format(current_epoch, batch_idx, i, text)
            gt_video_path = os.path.join(root, gt_video_filename)
            os.makedirs(os.path.split(gt_video_path)[0], exist_ok=True)
            subprocess.call(
                f"ffmpeg -i {gt_motion_path} -i {gt_music_path} -c copy {gt_video_path}", shell=True
            )

    @rank_zero_only
    def log_video(
        self, pl_module: LightningModule, batch: Any, batch_idx: int, split: str
    ) -> None:
        if hasattr(pl_module, "generate_sample") and \
                callable(pl_module.generate_sample) and \
                batch_idx % self.batch_freq == 0 and \
                self.max_videos_per_generation > 0:
            print(f'Log videos on epoch {pl_module.current_epoch}')

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            # on both stages, log music motion generation results
            if pl_module.stage == 'train_music_motion' or pl_module.stage == 'train_caption':
                with torch.no_grad():
                    music_gen, motion_gen, music_ref, motion_ref, text_prompt = pl_module.generate_sample(
                        batch,
                        duration=self.duration,
                        conditional_guidance_scale=self.conditional_guidance_scale
                    )

                N = min(music_gen.shape[0], self.max_videos_per_generation)
                # randomly pick N samples from the generated results
                idx = torch.LongTensor(random.sample(range(music_gen.shape[0]), N))
                music_gen = music_gen[idx]
                motion_gen = motion_gen[idx]
                music_ref = music_ref[idx]
                motion_ref = motion_ref[idx]
                tempt_ls = []
                for num in idx:
                    tempt_ls.append(text_prompt[num])

                # load VQVAE models
                pkg = torch.load(self.music_vqvae_path, map_location='cpu')
                cfg = OmegaConf.create(pkg['xp.cfg'])
                music_vqvae = get_compression_model(cfg)
                music_vqvae.load_state_dict(pkg['best_state'])

                motion_vqvae = instantiate_from_config(self.motion_vqvae_config.model)
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

                self.log_local(
                    pl_module.logger.save_dir, split, generated_music, generated_joint, ref_music, ref_joint,
                    pl_module.current_epoch, batch_idx
                )

            if pl_module.stage == 'train_caption':
                with torch.no_grad():
                    text_prompt, music_ref, motion_ref = pl_module.generate_captions(batch)

                N = min(music_ref.shape[0], self.max_videos_per_generation)
                # randomly pick N samples from the generated results
                idx = torch.LongTensor(random.sample(range(music_ref.shape[0]), N))
                music_ref = music_ref[idx]
                motion_ref = motion_ref[idx]
                tempt_ls = []
                for num in idx:
                    tempt_ls.append(text_prompt[num])
                text_prompt = tempt_ls

                # load VQVAE model
                pkg = torch.load(self.music_vqvae_path, map_location='cpu')
                cfg = OmegaConf.create(pkg['xp.cfg'])
                music_vqvae = get_compression_model(cfg)
                music_vqvae.load_state_dict(pkg['best_state'])

                motion_vqvae = instantiate_from_config(self.motion_vqvae_config.model)
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

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if not self.disabled and pl_module.current_epoch % self.epoch_freq == 0:
            self.log_video(pl_module, batch, batch_idx, split="train")

        if batch_idx == 0:
            model_size = torch.cuda.max_memory_allocated(device=None)
            for _ in range(30):
                model_size /= 2
            print('############### GPU memory used %.1f GB #################' % model_size)

    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Optional[STEP_OUTPUT],
        batch: Any, batch_idx: int, dataloader_idx: int = 0,
    ) -> None:
        if not self.disabled and pl_module.current_epoch % self.epoch_freq == 0:
            self.log_video(pl_module, batch, batch_idx, split="val")
