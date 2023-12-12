import typing as tp
import warnings

import flashy.distrib
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR
from mmldm.util import instantiate_from_config
from mmldm.audio.audiocraft_new.models.mm_lm_ablation import LMModel_Ablation1
from mmldm.audio.audiocraft_new.models.builders import get_debug_lm_model
from mmldm.audio.audiocraft_new.models.loaders import load_mm_lm_ablation_1_model
from mmldm.audio.audiocraft_new.modules.conditioners import ConditioningAttributes, WavCondition
from mmldm.audio.audiocraft_new.utils.autocast import TorchAutocast


MelodyList = tp.List[tp.Optional[torch.Tensor]]
MelodyType = tp.Union[torch.Tensor, MelodyList]


# backward compatible names mapping
_HF_MODEL_CHECKPOINTS_MAP = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
    "melody": "facebook/musicgen-melody",
}


class MM_Transformer(pl.LightningModule):
    def __init__(self,
                 name: str,
                 music_key: str,
                 motion_key: str,
                 text_cond_key: str,
                 motion_weight: float,
                 length_single_modal: int,
                 text_model_config: dict,
                 frame_rate: int = 50,
                 max_duration: tp.Optional[float] = None,

                 stage: tp.Optional[str] = None,
                 mm_ckpt: tp.Optional[str] = None,

                 autocast: bool = True,
                 autocast_dtype: str = 'float16',

                 generation_params: tp.Optional[dict] = None,
                 scheduler_config: tp.Optional[dict] = None,
                 optimization_config: tp.Optional[dict] = None,

                 monitor=None):
        super().__init__()

        self.music_key = music_key
        self.motion_key = motion_key
        self.text_cond_key = text_cond_key

        self.motion_weight = motion_weight

        # load music motion transformer
        self.name = name
        self.model: LMModel_Ablation1 = self.get_pretrained_lm(name, use_autocast=autocast)

        # load music motion captioner
        self.text_model = instantiate_from_config(text_model_config)

        assert stage is None or stage in ['train_music_motion', 'train_caption']
        self.stage = stage
        if self.stage == 'train_music_motion':
            # freeze text model
            print('In training music motion stage!')
            for p in self.text_model.parameters():
                p.requires_grad = False
        if self.stage == 'train_caption':
            print('In training caption stage!')
            assert mm_ckpt is not None
            pretrained_sd = torch.load(mm_ckpt, map_location='cpu')['state_dict']
            model_sd = self.model.state_dict()

            # select weights that belongs to music motion model, and strip 'model.' from keys
            pretrained_sd = {k: v for k, v in pretrained_sd.items() if k[len('model.'):] in model_sd.keys()}
            pretrained_sd = {k[len('model.'):]: v for k, v in pretrained_sd.items()}
            for k in self.model.state_dict().keys():
                assert k in pretrained_sd.keys()
            self.model.load_state_dict(pretrained_sd)
            print(f'Music motion model load from state dict of size {len(pretrained_sd)}')

            # freeze music motion model
            for p in self.model.parameters():
                p.requires_grad = False

        self.extend_stride = generation_params.pop('extend_stride')
        self.duration = generation_params.pop('duration')
        self.frame_rate = frame_rate
        self.sample_rate = 32000
        self.generation_params = generation_params

        self.max_sequence_length = (length_single_modal + self.model.n_q) * 2

        self.scheduler_config = scheduler_config
        self.optimization_config = optimization_config

        if max_duration is None:
            if hasattr(self.model, 'cfg'):
                max_duration = self.model.cfg.dataset.segment_duration  # type: ignore
            else:
                raise ValueError("You must provide max_duration when building directly MusicGen")
        assert max_duration is not None
        self.max_duration: float = max_duration

        dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16}
        self.autocast = TorchAutocast(
            enabled=autocast, device_type='cuda', dtype=dtype_map[autocast_dtype])
        if autocast:
            print(f'Using auto cast to {dtype_map[autocast_dtype]}')
        else:
            print('Not using autocast!')

        self.scaler: tp.Optional[torch.cuda.amp.GradScaler] = None
        if autocast and autocast_dtype == 'float16':
            self.scaler = torch.cuda.amp.GradScaler()
            print(f'Using scaler')
        if self.scaler is None:
            print('Not using scaler!')

        # set to manual backward in training step
        self.automatic_optimization = False

        if monitor is not None:
            self.monitor = monitor

    def get_pretrained_lm(self, name: str = 'facebook/musicgen-melody', device=None, use_autocast=True):
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        print(f'Load lm and conditioner to {device}')

        if name == 'debug':
            # used only for unit tests
            lm = get_debug_lm_model(device)
            return lm

        if name in _HF_MODEL_CHECKPOINTS_MAP:
            warnings.warn(
                "MusicGen pretrained model relying on deprecated checkpoint mapping. " +
                f"Please use full pre-trained id instead: facebook/musicgen-{name}")
            name = _HF_MODEL_CHECKPOINTS_MAP[name]

        lm = load_mm_lm_ablation_1_model(name, device=device, use_autocast=use_autocast)
        if 'self_wav' in lm.condition_provider.conditioners:
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        return lm

    def training_step(self, batch, batch_idx):
        music_code, motion_code, text_cond = batch[self.music_key], batch[self.motion_key], batch[self.text_cond_key]

        if self.stage == 'train_music_motion':
            text_condition = self.prepare_text_condition(text_cond)

            with self.autocast:
                music_output, motion_output = self.model.compute_predictions(music_code, motion_code, [], condition_tensors=text_condition)
                music_logits = music_output.logits
                music_mask = music_output.mask
                motion_logits = motion_output.logits
                motion_mask = motion_output.mask

                music_loss, music_loss_per_codebook = self._compute_cross_entropy(music_logits, music_code, music_mask)
                motion_loss, motion_loss_per_codebook = self._compute_cross_entropy(motion_logits, motion_code, motion_mask)
                total_loss = music_loss * (1 - self.motion_weight) + motion_loss * self.motion_weight

            self.log("train/loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            self.log("train/music_loss", music_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            self.log("train/motion_loss", motion_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)

            log_dict = {}
            for k in range(len(music_loss_per_codebook)):
                log_dict[f'train/music_ce_q{k + 1}'] = music_loss_per_codebook[k]
                log_dict[f'train/motion_ce_q{k + 1}'] = motion_loss_per_codebook[k]

            optimizer = self.optimizers().optimizer
            lr_scheduler = self.lr_schedulers()

            if self.scaler is not None:
                total_loss = self.scaler.scale(total_loss)

            if self.optimization_config['eager_sync']:
                with flashy.distrib.eager_sync_model(self.model):
                    self.manual_backward(total_loss)
            else:
                self.manual_backward(total_loss)
                flashy.distrib.sync_model(self.model)

            if self.scaler is not None:
                self.scaler.unscale_(optimizer)
            if self.optimization_config['max_norm']:
               log_dict['grad_norm'] = torch.nn.utils.clip_grad_norm_(
                   self.model.parameters(), self.optimization_config['max_norm']
               )

            if self.scaler is None:
                optimizer.step()
            else:
                self.scaler.step(optimizer)
                self.scaler.update()

            if lr_scheduler is not None:
                lr_scheduler.step()

            optimizer.zero_grad()

            if self.scaler is not None:
                log_dict['grad_scale'] = self.scaler.get_scale()

            self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        elif self.stage == 'train_caption':
            batch_size = len(text_cond)
            descriptions: tp.List[tp.Optional[str]] = [None] * batch_size
            null_text_condition = self.prepare_text_condition(descriptions)  # use null condition

            with self.autocast:
                with torch.no_grad():
                    self.model.eval()
                    music_motion_context = self.model.get_music_motion_context(music_code, motion_code, [], condition_tensors=null_text_condition)

                text_loss = self.text_model(text_cond, music_motion_context)

            self.log("train/text_loss", text_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)

            optimizer = self.optimizers().optimizer
            lr_scheduler = self.lr_schedulers()

            self.manual_backward(text_loss)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()

        else:
            ValueError()

    def validation_step(self, batch, batch_idx):
        music_code, motion_code, text_cond = batch[self.music_key], batch[self.motion_key], batch[self.text_cond_key]

        if self.stage == 'train_music_motion':
            text_condition = self.prepare_text_condition(text_cond)

            with self.autocast:
                music_output, motion_output = self.model.compute_predictions(music_code, motion_code, [],
                                                                             condition_tensors=text_condition)
                music_logits = music_output.logits
                music_mask = music_output.mask
                motion_logits = motion_output.logits
                motion_mask = motion_output.mask

                music_loss, music_loss_per_codebook = self._compute_cross_entropy(music_logits, music_code, music_mask)
                motion_loss, motion_loss_per_codebook = self._compute_cross_entropy(motion_logits, motion_code, motion_mask)
                total_loss = music_loss * (1 - self.motion_weight) + motion_loss * self.motion_weight

            self.log("val/loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("val/music_loss", music_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("val/motion_loss", motion_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            log_dict = {}
            for k in range(len(music_loss_per_codebook)):
                log_dict[f'val/music_ce_q{k + 1}'] = music_loss_per_codebook[k]
                log_dict[f'val/motion_ce_q{k + 1}'] = motion_loss_per_codebook[k]

            self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        elif self.stage == 'train_caption':
            batch_size = len(text_cond)
            descriptions: tp.List[tp.Optional[str]] = [None] * batch_size
            null_text_condition = self.prepare_text_condition(descriptions)  # use null condition

            with self.autocast:
                with torch.no_grad():
                    self.model.eval()
                    music_motion_context = self.model.get_music_motion_context(music_code, motion_code, [], condition_tensors=null_text_condition)

                text_loss = self.text_model(text_cond, music_motion_context)

            self.log("val/text_loss", text_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    def _compute_cross_entropy(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        B, K, T = targets.shape
        assert logits.shape[:-1] == targets.shape
        assert mask.shape == targets.shape
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook: tp.List[torch.Tensor] = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            ce_targets = targets_k[mask_k]
            ce_logits = logits_k[mask_k]
            q_ce = F.cross_entropy(ce_logits, ce_targets)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        # average cross entropy across codebooks
        ce = ce / K
        return ce, ce_per_codebook

    def prepare_text_condition(self, descriptions: tp.List[str]) -> dict:
        attributes = [ConditioningAttributes(text={'description': description})
                      for description in descriptions]

        attributes = self.model.cfg_dropout(attributes)
        attributes = self.model.att_dropout(attributes)

        tokenized = self.model.condition_provider.tokenize(attributes, device=self.device)

        # skip generating padding mask as in MusicGen

        with self.autocast:
            condition_tensors = self.model.condition_provider(tokenized)

        return condition_tensors

    def generate_sample(
            self,
            batch: dict,
            duration: tp.Optional[float] = None,
            conditional_guidance_scale: tp.Optional[float] = None,
            temperature: tp.Optional[float] = None,
        ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tp.List[str]]:
        attributes, _ = self._prepare_tokens_and_attributes(batch[self.text_cond_key], None)

        music_gen, motion_gen = self._generate_tokens(attributes, duration=duration, prompt_tokens=None,
                                                      temperature=temperature,
                                                      conditional_guidance_scale=conditional_guidance_scale)

        return music_gen, motion_gen, batch[self.music_key], batch[self.motion_key], batch[self.text_cond_key]

    def generate_single_modality(
            self,
            music_code: tp.Optional[torch.LongTensor] = None,  # (B, K, S)
            motion_code: tp.Optional[torch.LongTensor] = None,  # (B, K, S)
            text_description: tp.Optional[tp.List[str]] = None,
            conditional_guidance_scale: tp.Optional[float] = None,
            temperature: tp.Optional[float] = None,
    ) -> torch.LongTensor:
        assert music_code is None ^ motion_code is None, "Only one modality should be given."
        batch_size = music_code.shape[0] if music_code is not None else motion_code.shape[0]
        sequence_length = music_code.shape[-1] if music_code is not None else motion_code.shape[-1]
        if text_description is None:
            text_description = [None] * batch_size

        duration = sequence_length / self.frame_rate

        attributes, _ = self._prepare_tokens_and_attributes(text_description, None)

        music_gen, motion_gen = self._generate_tokens(attributes, duration=duration, prompt_tokens=None,
                                                      music_code=music_code, motion_code=motion_code,
                                                      temperature=temperature,
                                                      conditional_guidance_scale=conditional_guidance_scale)
        if music_code is None and motion_code is not None:
            return music_gen
        elif motion_code is None and music_code is not None:
            return motion_gen
        else:
            ValueError()

    def generate_captions(self, batch: dict) -> tp.Tuple[tp.List[str], torch.LongTensor, torch.LongTensor]:
        music_code, motion_code, text_cond = batch[self.music_key], batch[self.motion_key], batch[self.text_cond_key]
        batch_size = len(text_cond)
        descriptions: tp.List[tp.Optional[str]] = [None] * batch_size
        null_text_condition = self.prepare_text_condition(descriptions)  # use null condition

        with self.autocast:
            music_motion_context = self.model.get_music_motion_context(music_code, motion_code, [], condition_tensors=null_text_condition)
            captions = self.text_model.generate_caption(music_motion_context)

        return captions, music_code, motion_code


    @torch.no_grad()
    def _prepare_tokens_and_attributes(
            self,
            descriptions: tp.Sequence[tp.Optional[str]],
            prompt: tp.Optional[torch.Tensor],
            melody_wavs: tp.Optional[MelodyList] = None,
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        """Prepare model inputs.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            prompt (torch.Tensor): A batch of waveforms used for continuation.
            melody_wavs (torch.Tensor, optional): A batch of waveforms
                used as melody conditioning. Defaults to None.
        """
        attributes = [
            ConditioningAttributes(text={'description': description})
            for description in descriptions]

        if melody_wavs is None:
            for attr in attributes:
                attr.wav['self_wav'] = WavCondition(
                    torch.zeros((1, 1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    sample_rate=[self.sample_rate],
                    path=[None])
        else:
            if 'self_wav' not in self.model.condition_provider.conditioners:
                raise RuntimeError("This model doesn't support melody conditioning. "
                                   "Use the `melody` model.")
            assert len(melody_wavs) == len(descriptions), \
                f"number of melody wavs must match number of descriptions! " \
                f"got melody len={len(melody_wavs)}, and descriptions len={len(descriptions)}"
            for attr, melody in zip(attributes, melody_wavs):
                if melody is None:
                    attr.wav['self_wav'] = WavCondition(
                        torch.zeros((1, 1, 1), device=self.device),
                        torch.tensor([0], device=self.device),
                        sample_rate=[self.sample_rate],
                        path=[None])
                else:
                    attr.wav['self_wav'] = WavCondition(
                        melody[None].to(device=self.device),
                        torch.tensor([melody.shape[-1]], device=self.device),
                        sample_rate=[self.sample_rate],
                        path=[None],
                    )

        if prompt is not None:
            if descriptions is not None:
                assert len(descriptions) == len(prompt), "Prompt and nb. descriptions doesn't match"
            prompt = prompt.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None
        return attributes, prompt_tokens

    def _generate_tokens(
            self,
            attributes: tp.List[ConditioningAttributes],
            prompt_tokens: tp.Optional[torch.Tensor],
            music_code: tp.Optional[torch.LongTensor] = None,
            motion_code: tp.Optional[torch.LongTensor] = None,
            duration: tp.Optional[float] = None,
            conditional_guidance_scale: tp.Optional[float] = None,
            temperature: float = 1.
    ) -> tp.Tuple[torch.LongTensor, torch.LongTensor]:
        duration = self.duration if duration is None else duration
        total_gen_len = int(duration * self.frame_rate)
        max_prompt_len = int(min(duration, self.max_duration) * self.frame_rate)
        current_gen_offset: int = 0

        if prompt_tokens is not None:
            assert max_prompt_len >= prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"

        if duration <= self.max_duration:
            # generate by sampling from LM, simple case.
            with self.autocast:
                gen_tokens = self.model.generate(
                    prompt_tokens, attributes,
                    music_code=music_code,
                    motion_code=motion_code,
                    max_gen_len=total_gen_len,
                    use_sampling=self.generation_params['use_sampling'],
                    temp=self.generation_params['temp'] if temperature is None else temperature,
                    top_k=self.generation_params['top_k'],
                    top_p=self.generation_params['top_p'],
                    cfg_coef=self.generation_params['cfg_coef'] if conditional_guidance_scale is None else conditional_guidance_scale,
                    two_step_cfg=self.generation_params['two_step_cfg'],
                )

        return gen_tokens

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.optimization_config['learning_rate'],
            betas=self.optimization_config['betas'],
            weight_decay=self.optimization_config['weight_decay'],
            eps=self.optimization_config['eps']
        )

        if self.scheduler_config is None:
            return opt

        scheduler = instantiate_from_config(self.scheduler_config)
        print("Setting up LambdaLR scheduler...")
        scheduler = [
            {
                'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                'interval': 'step',
                'frequency': 1
            }]

        return [opt], scheduler
