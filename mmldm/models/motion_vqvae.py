from mmldm.util import instantiate_from_config
import torch
from omegaconf import OmegaConf
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange

from mmldm.audio.audiocraft.models.builders import get_compression_model
from mmldm.audio.audiocraft.quantization.vq import ResidualVectorQuantizer
from mmldm.motion.motion_process import recover_from_ric
from mmldm.modules.motion_vqvae_module import Encoder, Decoder


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class MM_VQVAE(pl.LightningModule):
    def __init__(self,
                 music_ddconfig,
                 motion_ddconfig,
                 pre_post_quantize_ddconfig,
                 lossconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 music_key="waveform",
                 motion_key="motion",
                 monitor=None,
                 ):
        super().__init__()
        self.motion_key = motion_key
        self.music_key = music_key

        self.music_encoder, self.quantizer = self.instantiate_music_vqvae(**music_ddconfig)

        self.motion_encoder = Encoder(**motion_ddconfig)
        self.motion_decoder = Decoder(**motion_ddconfig)

        # instantiate new codebook
        joint_dimension = 128 + motion_ddconfig['output_dim']

        # instantiate the modules before quantizer
        pre_quant_conv_mult = pre_post_quantize_ddconfig['pre_quant_conv_mult']
        self.pre_quantize_conv = nn.Sequential(
            nn.Conv1d(joint_dimension, joint_dimension, 1),
            nn.ELU(),
            nn.Conv1d(joint_dimension, joint_dimension * pre_quant_conv_mult, 3, 1, 1),
            nn.ELU(),
            nn.Conv1d(joint_dimension * pre_quant_conv_mult, joint_dimension, 3, 1, 1),
            nn.ELU(),
            nn.Conv1d(joint_dimension, motion_ddconfig['output_dim'], 1)
        )

        # instantiate the modules after quantizer
        post_quant_conv_mult = pre_post_quantize_ddconfig['post_quant_conv_mult']
        self.post_quantize_conv = nn.Sequential(
            nn.Conv1d(joint_dimension, joint_dimension, 1),
            nn.ELU(),
            nn.Conv1d(joint_dimension, joint_dimension * post_quant_conv_mult, 3, 1, 1),
            nn.ELU(),
            nn.Conv1d(joint_dimension * post_quant_conv_mult, joint_dimension, 3, 1, 1),
            nn.ELU(),
            nn.Conv1d(joint_dimension, motion_ddconfig['output_dim'], 1)
        )

        self.loss = instantiate_from_config(lossconfig)

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def instantiate_music_vqvae(self, vqvae_ckpt):
        pkg = torch.load(vqvae_ckpt, map_location='cpu')
        cfg = OmegaConf.create(pkg['xp.cfg'])
        model = get_compression_model(cfg)
        model.load_state_dict(pkg['best_state'])

        encoder = model.encoder.to(self.device)
        quantizer = model.quantizer.to(self.device)

        for p in encoder.parameters():
            p.requires_grad = False

        quantizer.freeze_codebook = True

        return encoder, quantizer

    def encode(self, inputs):
        x_motion = inputs['motion']  # B, T1, D1
        x_music = inputs['music']

        assert x_music.dim() == 3
        with torch.no_grad():
            music_emb = self.music_encoder(x_music)  # B, 128, T

        assert x_motion.dim() == 3
        x_motion = rearrange(x_motion, 'b t d -> b d t')
        motion_emb = self.motion_encoder(x_motion)  # B, 128, T

        # pre quant residual module
        x_catted = torch.cat((music_emb, motion_emb), dim=1)  # B, 256, T
        ff_emb = self.pre_quantize_conv(x_catted)
        motion_emb = motion_emb + ff_emb  # B, 128, T

        return music_emb, motion_emb

    def encode_with_music_token(self, x_motion, music_emb):  # x_motion: B, T, 263
        assert x_motion.dim() == 3
        x_motion = rearrange(x_motion, 'b t d -> b d t')
        motion_emb = self.motion_encoder(x_motion)  # B, 128, T

        # pre quant residual module
        x_catted = torch.cat((music_emb, motion_emb), dim=1)  # B, 256, T
        ff_emb = self.pre_quantize_conv(x_catted)
        motion_emb = motion_emb + ff_emb  # B, 128, T

        # quantize
        q_res_music = self.quantizer(music_emb, 50)  # 50 is the fixed sample rate
        q_res_motion = self.quantizer(motion_emb, 50)

        return q_res_music.x, q_res_motion.x

    def decode(self, music_emb, motion_emb):
        # post quant residual module
        x_catted = torch.cat((music_emb, motion_emb), dim=1)
        ff_emb = self.post_quantize_conv(x_catted)
        motion_emb = motion_emb + ff_emb

        motion_recon = self.motion_decoder(motion_emb)
        motion_recon = rearrange(motion_recon, 'b d t -> b t d')

        return motion_recon

    def decode_from_code(self, music_code, motion_code):
        music_emb = self.quantizer.decode(music_code)
        motion_emb = self.quantizer.decode(motion_code)
        return self.decode(music_emb, motion_emb)

    def forward(self, input):
        music_emb, motion_emb = self.encode(input)

        q_res_music = self.quantizer(music_emb, 50)  # 50 is the fixed sample rate
        q_res_motion = self.quantizer(motion_emb, 50)

        motion_recon = self.decode(q_res_music.x, q_res_motion.x)

        return motion_recon, q_res_motion.penalty  # penalty is the commitment loss

    @torch.no_grad()
    def encode_music_first_stage(self, x):
        return self.music_vae_model.encode(x)  # !!! not modified yet

    def motion_vec_to_joint(self, vec, motion_mean, motion_std):
        # vec: [bs, 200, 263]
        mean = torch.tensor(motion_mean).to(vec)
        std = torch.tensor(motion_std).to(vec)
        vec = vec*std + mean
        joint = recover_from_ric(vec, joints_num=22)
        joint = joint.cpu().detach().numpy()
        return joint

    def get_input(self, batch, music_k, motion_k, bs=None):
        motion = batch[motion_k]
        x_motion = motion.to(memory_format=torch.contiguous_format).float()
        waveform = batch[music_k]
        x_music = waveform.unsqueeze(1).to(memory_format=torch.contiguous_format).float()
        if bs is not None:
            x_motion = x_motion[:bs]
            x_music = x_music[:bs]
        x_music = x_music.to(self.device)
        x_motion = x_motion.to(self.device) # B, 200, 263
        return {'motion': x_motion, 'music': x_music}

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.music_key, self.motion_key)
        motion_recon, commitment_loss = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, motion_recon, commitment_loss, split="train")

        print(torch.mean(self.quantizer.vq.layers[0]._codebook.embed[0]))
        if batch_idx % 20 == 0:
            print(self.quantizer.vq.layers[0]._codebook.embed[0])

        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return aeloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.music_key, self.motion_key)
        motion_recon, commitment_loss = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, motion_recon, commitment_loss, split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0)
        return [opt_ae], []

    @torch.no_grad()
    def log_videos(self, batch, motion_mean=None, motion_std=None):
        inputs = self.get_input(batch, self.music_key, self.motion_key)
        motion_recon, _ = self(inputs)
        gt_waveform = batch[self.music_key].unsqueeze(1).detach().cpu().numpy()
        waveform = gt_waveform

        joint = self.motion_vec_to_joint(motion_recon, motion_mean, motion_std)
        gt_joint = self.motion_vec_to_joint(batch[self.motion_key], motion_mean, motion_std)
        return waveform, joint, gt_waveform, gt_joint


if __name__ == "__main__":
    lossconfig = {
        'target': 'mmldm.modules.losses.mm_vqvae_loss.MusicMotion_VQVAE_Loss_v2',
        'params': {'commitment_loss_weight': 0.02}
    }

    motion_ddconfig = {
        'input_dim': 263,
        'output_dim': 16,
        'emb_dim_encoder': [256, 128, 32, 16],
        'emb_dim_decoder': [16, 32, 128, 256],
        'input_fps': 20,
        'rvq_fps': 50,
        'dilation_growth_rate': 2,
        'depth_per_res_block': 1,
        'activation': 'relu'
    }

    music_ddconfig = {
        'vqvae_ckpt': 'C:\\Users\\Mahlering\\.cache\\huggingface\\hub\\models--facebook--musicgen-small\\snapshots\\bf842007f70cf1b174daa4f42b5e45a21d59d3ec\\compression_state_dict.bin',
        'use_pretrained': True,
        'freeze_encoder': True,
        'freeze_decoder': True,
        'freeze_codebook': True
    }

    codebook_ddconfig = {
        'n_q': 4,
        'q_dropout': False,
        'bins': 2048,
        'decay': 0.99,
        'kmeans_init': False,
        'kmeans_iters': 50,
        'threshold_ema_dead_code': 2,
        'orthogonal_reg_weight': 0.0,
        'orthogonal_reg_active_codes_only': False
    }

    pre_post_quantize_ddconfig = {
        'pre_quant_conv_mult': 4,
        'post_quant_conv_mult': 4,
    }

    model = MM_VQVAE(
        music_ddconfig, motion_ddconfig, codebook_ddconfig,
        pre_post_quantize_ddconfig, lossconfig
    )

    music = torch.randn((3, 1, 32000 * 10))
    motion = torch.randn((3, 200, 263))

    inputs = {'music': music, 'motion': motion}

    #motion_recon, music_recon, commitment_loss = model(inputs)

    print('here')



