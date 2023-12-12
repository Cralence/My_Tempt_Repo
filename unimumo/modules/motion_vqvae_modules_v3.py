import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None):
        super().__init__()
        padding = dilation
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)

        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()

        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0, )

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        super().__init__()

        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm)
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self,
                 input_dim=263,
                 output_dim=16,
                 emb_dim_encoder=[256, 192, 128, 64, 32, 16],
                 input_fps=20,
                 rvq_fps=50,
                 dilation_growth_rate=2,
                 depth_per_res_block=6,
                 activation='relu',
                 norm=None,
                 **kwargs):
        super().__init__()

        self.input_fps = input_fps
        self.rvq_fps = rvq_fps

        self.init_conv = nn.Sequential(
            nn.Conv1d(input_dim, emb_dim_encoder[0], 3, 1, 1),
            nn.ReLU()
        )

        blocks = []

        for i in range(len(emb_dim_encoder) - 1):
            in_channel = emb_dim_encoder[i]
            out_channel = emb_dim_encoder[i + 1]

            block = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, 3, 1, 1),
                Resnet1D(out_channel, n_depth=depth_per_res_block, dilation_growth_rate=dilation_growth_rate,
                         activation=activation, norm=norm)
            )
            blocks.append(block)

        self.resnet_model = nn.Sequential(*blocks)

        self.post_conv = nn.Conv1d(emb_dim_encoder[-1], output_dim, 3, 1, 1)

    def forward(self, x):
        input_T = x.shape[-1]
        target_T = int(input_T / self.input_fps * self.rvq_fps)
        x = nn.functional.interpolate(x, size=target_T, mode='linear')

        x = self.init_conv(x)
        x = self.resnet_model(x)
        x = self.post_conv(x)

        return x


class Decoder(nn.Module):
    def __init__(self,
                 input_dim=263,
                 output_dim=16,
                 emb_dim_decoder=[16, 32, 64, 128, 192, 256],
                 input_fps=20,
                 rvq_fps=50,
                 dilation_growth_rate=2,
                 depth_per_res_block=6,
                 activation='relu',
                 norm=None,
                 **kwargs):
        super().__init__()

        self.input_fps = input_fps
        self.rvq_fps = rvq_fps

        self.init_conv = nn.Sequential(
            nn.Conv1d(output_dim, emb_dim_decoder[0], 3, 1, 1),
            nn.ReLU()
        )

        blocks = []
        for i in range(len(emb_dim_decoder) - 1):
            in_channel = emb_dim_decoder[i]
            out_channel = emb_dim_decoder[i + 1]

            block = nn.Sequential(
                Resnet1D(in_channel, depth_per_res_block,
                         dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Conv1d(in_channel, out_channel, 3, 1, 1)
            )
            blocks.append(block)
        self.resnet_block = nn.Sequential(*blocks)

        self.post_conv = nn.Sequential(
            nn.Conv1d(emb_dim_decoder[-1], emb_dim_decoder[-1], 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(emb_dim_decoder[-1], input_dim, 3, 1, 1)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.resnet_block(x)

        current_T = x.shape[-1]
        target_T = int(current_T / self.rvq_fps * self.input_fps)
        x = nn.functional.interpolate(x, target_T, mode='linear')

        x = self.post_conv(x)

        return x


class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, mu):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.reset_codebook()

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):

        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                                  keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()  # (N, DIM, T)

        return x_d, commit_loss, perplexity



if __name__ == '__main__':
    encoder = Encoder()
    decoder = Decoder()
    print('encoder: ', sum(p.numel() for p in encoder.parameters()), f', decoder: {sum(p.numel() for p in decoder.parameters())}')

    x = torch.randn(3, 263, 200)
    emb = encoder(x)
    print(emb.shape)
    recon = decoder(emb)
    print(recon.shape)


