import torch
import torch.nn as nn

class MusicMotion_VQVAE_Loss(nn.Module):
    def __init__(self, commitment_loss_weight=1.0, music_weight=1.0, motion_weight=1.0):

        super().__init__()
        self.commitment_loss_weight = commitment_loss_weight
        self.music_weight = music_weight
        self.motion_weight = motion_weight
        self.recon_loss = nn.MSELoss()

    def forward(self, inputs, motion_recon, music_recon, commitment_loss, dist_ref=None, split="train"):
        music_gt = inputs['music']
        motion_gt = inputs['motion']
        motion_rec_loss = self.recon_loss(motion_recon.contiguous(), motion_gt.contiguous())
        music_rec_loss = self.recon_loss(music_recon.contiguous(), music_gt.contiguous())

        loss = self.motion_weight * motion_rec_loss + self.music_weight * music_rec_loss + self.commitment_loss_weight * commitment_loss
        rec_loss = self.motion_weight * motion_rec_loss + self.music_weight * music_rec_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/commitment_loss".format(split): commitment_loss.detach().mean(),
               "{}/music_rec_loss".format(split): music_rec_loss.detach().mean(),
               "{}/motion_rec_loss".format(split): motion_rec_loss.detach().mean(),
               "{}/rec_loss".format(split): rec_loss.detach().mean(),
               }
        return loss, log


class MusicMotion_VQVAE_Loss_v2(nn.Module):
    def __init__(self, commitment_loss_weight=1.0, music_weight=1.0, motion_weight=1.0):

        super().__init__()
        self.commitment_loss_weight = commitment_loss_weight
        self.music_weight = music_weight
        self.motion_weight = motion_weight
        self.recon_loss = nn.MSELoss()

    def forward(self, inputs, motion_recon, music_recon, commitment_loss, split="train"):
        music_gt = inputs['music']
        motion_gt = inputs['motion']
        motion_rec_loss = self.recon_loss(motion_recon.contiguous(), motion_gt.contiguous())
        music_rec_loss = self.recon_loss(music_recon.contiguous(), music_gt.contiguous())

        loss = self.motion_weight * motion_rec_loss + self.commitment_loss_weight * commitment_loss
        rec_loss = self.motion_weight * motion_rec_loss + self.music_weight * music_rec_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/commitment_loss".format(split): commitment_loss.detach().mean(),
               "{}/music_rec_loss".format(split): music_rec_loss.detach().mean(),
               "{}/motion_rec_loss".format(split): motion_rec_loss.detach().mean(),
               "{}/rec_loss".format(split): rec_loss.detach().mean(),
               }
        return loss, log


class MusicMotion_VQVAE_Loss_v3(nn.Module):
    def __init__(self, commitment_loss_weight=1.0, motion_weight=1.0):

        super().__init__()
        self.commitment_loss_weight = commitment_loss_weight
        self.motion_weight = motion_weight
        self.recon_loss = nn.MSELoss()

    def forward(self, inputs, motion_recon, commitment_loss, split="train"):
        motion_gt = inputs['motion']
        motion_rec_loss = self.recon_loss(motion_recon.contiguous(), motion_gt.contiguous())

        loss = self.motion_weight * motion_rec_loss + self.commitment_loss_weight * commitment_loss
        rec_loss = self.motion_weight * motion_rec_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/commitment_loss".format(split): commitment_loss.detach().mean(),
               "{}/rec_loss".format(split): rec_loss.detach().mean(),
               }
        return loss, log

class MusicMotion_VQVAE_Loss_v4(nn.Module):
    def __init__(self, commitment_loss_weight=1.0, motion_weight=1.0):

        super().__init__()
        self.commitment_loss_weight = commitment_loss_weight
        self.motion_weight = motion_weight
        self.recon_loss = nn.SmoothL1Loss()

    def forward(self, feature_recon, feature_gt, joint_recon, joint_gt, commitment_loss, split="train"):
        feature_recon_loss = self.recon_loss(feature_recon.contiguous(), feature_gt.contiguous())
        joint_recon_loss = self.recon_loss(joint_recon.contiguous(), joint_gt.contiguous())

        loss = self.motion_weight * feature_recon_loss + self.motion_weight * joint_recon_loss + self.commitment_loss_weight * commitment_loss
        rec_loss = self.motion_weight * feature_recon_loss + self.motion_weight * joint_recon_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/commitment_loss".format(split): commitment_loss.detach().mean(),
               "{}/rec_loss".format(split): rec_loss.detach().mean(),
               "{}/feature_loss".format(split): feature_recon_loss.detach().mean(),
               "{}/joint_loss".format(split): joint_recon_loss.detach().mean(),
               }
        return loss, log


class MusicMotion_VQVAE_Loss_v5(nn.Module):
    def __init__(self, recons_loss, nb_joints, weight_commit, weight_vel):
        super().__init__()

        if recons_loss == 'l1':
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2':
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss()

        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4

        self.weight_commit = weight_commit
        self.weight_vel = weight_vel

    def forward(self, motion_pred, motion_gt, commitment_loss, split="train"):
        recon_loss = self.Loss(motion_pred[..., : self.motion_dim], motion_gt[..., :self.motion_dim])
        vel_loss = self.Loss(motion_pred[..., 4: (self.nb_joints - 1) * 3 + 4],
                         motion_gt[..., 4: (self.nb_joints - 1) * 3 + 4])
        loss = recon_loss + self.weight_commit * commitment_loss + self.weight_vel * vel_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/commitment_loss".format(split): commitment_loss.detach().mean(),
               "{}/rec_loss".format(split): recon_loss.detach().mean(),
               "{}/vel_loss".format(split): vel_loss.detach().mean(),
               }

        return loss, log
