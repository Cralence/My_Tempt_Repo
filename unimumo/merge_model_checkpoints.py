import os
import argparse
from omegaconf import OmegaConf

import torch

# merge the checkpoints and configs in the three stages into a unified checkpoint
# that can be loaded in one time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        required=False,
        help="The directory to save the final checkpoint",
        default="./unimumo_checkpoint",
    )
    parser.add_argument(
        "--music_vqvae",
        type=str,
        required=False,
        default=None,
        help="The path of pretrained Encodec",
    )
    parser.add_argument(
        "--motion_vqvae_ckpt",
        type=str,
        required=False,
        default=None,
        help="The path of pretrained motion vqvae",
    )
    parser.add_argument(
        "--motion_vqvae_config",
        type=str,
        required=False,
        default='./configs/train_motion_vqvae.yaml',
        help="The path of motion vqvae configs",
    )
    parser.add_argument(
        "--mm_lm_ckpt",
        type=str,
        required=False,
        default=None,
        help="The path of pretrained music motion lm",
    )
    parser.add_argument(
        "--mm_lm_config",
        type=str,
        required=False,
        default='./configs/train_lm.yaml',
        help="The path of music motion lm configs",
    )

    args = parser.parse_args()

    assert os.path.exists(args.music_vqvae)
    assert os.path.exists(args.motion_vqvae_ckpt)
    assert os.path.exists(args.motion_vqvae_config)
    assert os.path.exists(args.mm_lm_ckpt)
    assert os.path.exists(args.mm_lm_config)
    os.makedirs(args.save_dir, exist_ok=True)

    unimumo_state_dict = {}

    encodec_weight = torch.load(args.music_vqvae, map_location='cpu')
    unimumo_state_dict['music_vqvae_config'] = encodec_weight['xp.cfg']  # omegaconf.DictConfig
    unimumo_state_dict['music_vqvae_weight'] = encodec_weight['best_state']  # dict[str, tensor]

    motion_vqvae_config = OmegaConf.load(args.motion_vqvae_config)
    unimumo_state_dict['motion_vqvae_config'] = motion_vqvae_config  # omegaconf.DictConfig

    motion_vqvae_weight = torch.load(args.motion_vqvae_ckpt, map_location='cpu')
    unimumo_state_dict['motion_vqvae_weight'] = motion_vqvae_weight  # dict[str, tensor]

    mm_lm_config = OmegaConf.load(args.mm_lm_config)
    unimumo_state_dict['mm_lm_config'] = mm_lm_config  # omegaconf.DictConfig

    mm_lm_ckpt = torch.load(args.mm_lm_ckpt, map_location='cpu')
    unimumo_state_dict['mm_lm_ckpt'] = mm_lm_ckpt  # dict[str, tensor]

    torch.save(unimumo_state_dict, os.path.join(args.save_dir, 'unimumo_model.ckpt'))

