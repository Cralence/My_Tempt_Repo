import typing as tp
import warnings
import flashy.distrib
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from unimumo.util import instantiate_from_config
from unimumo.audio.audiocraft_.models.mm_lm import LMModel, ConditionTensors
from unimumo.audio.audiocraft_.models.builders import get_debug_lm_model
from unimumo.audio.audiocraft_.models.loaders import load_mm_lm_model
from unimumo.audio.audiocraft_.modules.conditioners import ConditioningAttributes, WavCondition


MelodyList = tp.List[tp.Optional[torch.Tensor]]
MelodyType = tp.Union[torch.Tensor, MelodyList]


# backward compatible names mapping
_HF_MODEL_CHECKPOINTS_MAP = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
    "melody": "facebook/musicgen-melody",
}


class MusicMotionTransformer(pl.LightningModule):
    def __init__(
        self,
        name: str,
        music_key: str,
        motion_key: str,
        text_cond_key: str,
        motion_weight: float,
        length_single_modal: int,
        text_model_config: dict,
        feature_frame_rate: int = 50,

        stage: tp.Optional[str] = None,
        mm_ckpt: tp.Optional[str] = None,

        generation_params: tp.Optional[dict] = None,
        scheduler_config: tp.Optional[dict] = None,
        optimization_config: tp.Optional[dict] = None,

        monitor=None
    ):
        super().__init__()

        self.music_key = music_key
        self.motion_key = motion_key
        self.text_cond_key = text_cond_key

        self.motion_weight = motion_weight

        # load music motion transformer
        self.model: LMModel = self.get_pretrained_lm(name, use_autocast=False)

        # load music motion captioner
        self.text_model = instantiate_from_config(text_model_config)

        assert stage is None or stage in ['train_music_motion', 'train_caption']
        self.stage = stage
        if self.stage == 'train_music_motion':
            print('In training music motion stage!')
            # freeze text model
            for p in self.text_model.parameters():
                p.requires_grad = False
        if self.stage == 'train_caption':
            print('In training caption stage!')
            assert mm_ckpt is not None, "The pretrained music motion model is not provided"
            # load the music motion lm part of the ckpt
            pretrained_sd = torch.load(mm_ckpt, map_location='cpu')['state_dict']
            mm_lm_sd = {k: v for k, v in pretrained_sd.items() if k.startswith("model.")}  # find keys with prefix "model."
            mm_lm_sd = {k[len("model."):]: v for k, v in mm_lm_sd.items()}  # remove the prefix "model."
            self.model.load_state_dict(mm_lm_sd)
            # freeze music motion model
            for p in self.model.parameters():
                p.requires_grad = False

        self.duration = generation_params.pop('duration')
        self.feature_frame_rate = feature_frame_rate
        self.sample_rate = 32000
        self.generation_params = generation_params

        self.max_sequence_length = (length_single_modal + self.model.n_q) * 2

        self.scheduler_config = scheduler_config
        self.optimization_config = optimization_config

        # set to manual backward in training step
        self.automatic_optimization = False

        if monitor is not None:
            self.monitor = monitor

    def get_pretrained_lm(
        self,
        name: str = 'facebook/musicgen-melody',
        device=None, use_autocast=False
    ) -> LMModel:
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

        lm = load_mm_lm_model(name, device=device, use_autocast=use_autocast)
        if 'self_wav' in lm.condition_provider.conditioners:
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        return lm

    def training_step(
        self,
        batch: tp.Dict[str, tp.Union[torch.LongTensor, tp.List[str]]],
        batch_idx: int
    ):
        music_code, motion_code, text_cond = batch[self.music_key], batch[self.motion_key], batch[self.text_cond_key]

        if self.stage == 'train_music_motion':  # train the music motion lm
            text_condition = self.prepare_text_condition(text_cond)

            music_output, motion_output = self.model.compute_predictions(
                music_code, motion_code, [], condition_tensors=text_condition
            )
            music_logits, music_mask = music_output.logits, music_output.mask
            motion_logits, motion_mask = motion_output.logits, motion_output.mask

            music_loss, music_loss_per_codebook = self.compute_cross_entropy(music_logits, music_code, music_mask)
            motion_loss, motion_loss_per_codebook = self.compute_cross_entropy(motion_logits, motion_code, motion_mask)
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

            if self.optimization_config['eager_sync']:
                with flashy.distrib.eager_sync_model(self.model):
                    self.manual_backward(total_loss)
            else:
                self.manual_backward(total_loss)
                flashy.distrib.sync_model(self.model)

            if self.optimization_config['max_norm']:
                log_dict['grad_norm'] = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.optimization_config['max_norm']
                )

            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()

            self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        else:  # train the text generation model
            batch_size = len(text_cond)

            # use null condition for music motion network
            descriptions: tp.List[tp.Optional[str]] = [None] * batch_size
            null_text_condition = self.prepare_text_condition(descriptions)

            # get music motion features using music motion LM
            with torch.no_grad():
                self.model.eval()
                music_motion_context = self.model.get_music_motion_context(
                    music_code, motion_code, [], condition_tensors=null_text_condition
                )

            text_loss = self.text_model(text_cond, music_motion_context)

            self.log("train/text_loss", text_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)

            optimizer = self.optimizers().optimizer
            lr_scheduler = self.lr_schedulers()

            self.manual_backward(text_loss)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()

    def validation_step(
        self,
        batch: tp.Dict[str, tp.Union[torch.LongTensor, tp.List[str]]],
        batch_idx: int
    ):
        music_code, motion_code, text_cond = batch[self.music_key], batch[self.motion_key], batch[self.text_cond_key]

        if self.stage == 'train_music_motion':
            text_condition = self.prepare_text_condition(text_cond)

            music_output, motion_output = self.model.compute_predictions(
                music_code, motion_code, [], condition_tensors=text_condition
            )
            music_logits, music_mask = music_output.logits, music_output.mask
            motion_logits, motion_mask = motion_output.logits, motion_output.mask

            music_loss, music_loss_per_codebook = self.compute_cross_entropy(music_logits, music_code, music_mask)
            motion_loss, motion_loss_per_codebook = self.compute_cross_entropy(motion_logits, motion_code, motion_mask)
            total_loss = music_loss * (1 - self.motion_weight) + motion_loss * self.motion_weight

            self.log("val/loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("val/music_loss", music_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("val/motion_loss", motion_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            log_dict = {}
            for k in range(len(music_loss_per_codebook)):
                log_dict[f'val/music_ce_q{k + 1}'] = music_loss_per_codebook[k]
                log_dict[f'val/motion_ce_q{k + 1}'] = motion_loss_per_codebook[k]

            self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        else:
            batch_size = len(text_cond)

            # use null condition for music motion network
            descriptions: tp.List[tp.Optional[str]] = [None] * batch_size
            null_text_condition = self.prepare_text_condition(descriptions)

            # get music motion features using music motion LM
            with torch.no_grad():
                self.model.eval()
                music_motion_context = self.model.get_music_motion_context(
                    music_code, motion_code, [], condition_tensors=null_text_condition
                )

            text_loss = self.text_model(text_cond, music_motion_context)

            self.log("val/text_loss", text_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    def compute_cross_entropy(
        self, logits: torch.Tensor, targets: torch.LongTensor, mask: torch.Tensor
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

    def prepare_text_condition(self, descriptions: tp.List[str]) -> ConditionTensors:
        attributes = [ConditioningAttributes(text={'description': description}) for description in descriptions]

        attributes = self.model.cfg_dropout(attributes)
        attributes = self.model.att_dropout(attributes)

        tokenized = self.model.condition_provider.tokenize(attributes, device=self.device)
        condition_tensors = self.model.condition_provider(tokenized)

        return condition_tensors

    def generate_sample(
        self,
        batch: tp.Dict[str, tp.Union[torch.LongTensor, tp.List[str]]],
        duration: tp.Optional[float] = None,
        conditional_guidance_scale: tp.Optional[float] = None,
        temperature: tp.Optional[float] = None,
        return_result_only: bool = False
    ):
        attributes = self._prepare_tokens_and_attributes(batch[self.text_cond_key])

        music_gen, motion_gen = self._generate_tokens(
            attributes, duration=duration, temperature=temperature,
            conditional_guidance_scale=conditional_guidance_scale
        )
        if return_result_only:
            return music_gen, motion_gen
        else:
            return music_gen, motion_gen, batch[self.music_key], batch[self.motion_key], batch[self.text_cond_key]

    def generate_single_modality(
        self,
        music_code: tp.Optional[torch.LongTensor] = None,  # (B, K, S)
        motion_code: tp.Optional[torch.LongTensor] = None,  # (B, K, S)
        text_description: tp.Optional[tp.List[str]] = None,
        conditional_guidance_scale: tp.Optional[float] = None,
        temperature: tp.Optional[float] = None,
    ) -> torch.LongTensor:
        assert (music_code is None) ^ (motion_code is None), "Only one modality should be given."
        batch_size = music_code.shape[0] if music_code is not None else motion_code.shape[0]
        sequence_length = music_code.shape[-1] if music_code is not None else motion_code.shape[-1]
        if text_description is None:
            text_description = [None] * batch_size

        duration = sequence_length / self.feature_frame_rate

        attributes = self._prepare_tokens_and_attributes(text_description)

        music_gen, motion_gen = self._generate_tokens(
            attributes, duration=duration, music_code=music_code, motion_code=motion_code,
            temperature=temperature, conditional_guidance_scale=conditional_guidance_scale
        )
        if music_code is None and motion_code is not None:
            return music_gen
        else:
            return motion_gen

    def generate_captions(
        self,
        batch: tp.Dict[str, tp.Union[torch.LongTensor, tp.List[str]]],
        return_caption_only: bool = False
    ) -> tp.Union[tp.List[str], tp.Tuple[tp.List[str], torch.LongTensor, torch.LongTensor]]:
        music_code, motion_code, text_cond = batch[self.music_key], batch[self.motion_key], batch[self.text_cond_key]
        batch_size = len(text_cond)
        descriptions: tp.List[tp.Optional[str]] = [None] * batch_size
        null_text_condition = self.prepare_text_condition(descriptions)  # use null condition

        music_motion_context = self.model.get_music_motion_context(
            music_code, motion_code, [], condition_tensors=null_text_condition
        )
        captions = self.text_model.generate_caption(music_motion_context)

        if return_caption_only:
            return captions
        else:
            return captions, music_code, motion_code

    @torch.no_grad()
    def _prepare_tokens_and_attributes(
        self,
        descriptions: tp.Sequence[tp.Optional[str]],
    ) -> tp.List[ConditioningAttributes]:
        attributes = [ConditioningAttributes(text={'description': description}) for description in descriptions]

        for attr in attributes:
            attr.wav['self_wav'] = WavCondition(
                torch.zeros((1, 1, 1), device=self.device),
                torch.tensor([0], device=self.device),
                sample_rate=[self.sample_rate],
                path=[None])

        return attributes

    def _generate_tokens(
        self,
        attributes: tp.List[ConditioningAttributes],
        music_code: tp.Optional[torch.LongTensor] = None,
        motion_code: tp.Optional[torch.LongTensor] = None,
        duration: tp.Optional[float] = None,
        conditional_guidance_scale: tp.Optional[float] = None,
        temperature: float = 1.
    ) -> tp.Tuple[torch.LongTensor, torch.LongTensor]:
        duration = self.duration if duration is None else duration
        total_gen_len = int(duration * self.feature_frame_rate)

        # generate by sampling from LM
        gen_tokens = self.model.generate(
            attributes,
            music_code=music_code,
            motion_code=motion_code,
            max_gen_len=total_gen_len,
            use_sampling=self.generation_params['use_sampling'],
            temp=self.generation_params['temp'] if temperature is None else temperature,
            top_k=self.generation_params['top_k'],
            top_p=self.generation_params['top_p'],
            cfg_coef=self.generation_params['cfg_coef'] if conditional_guidance_scale is None else conditional_guidance_scale,
        )

        return gen_tokens

    def configure_optimizers(self):
        opt = None
        if self.stage == 'train_music_motion':
            opt = torch.optim.AdamW(
                params=self.model.parameters(),
                lr=self.optimization_config['learning_rate'],
                betas=self.optimization_config['betas'],
                weight_decay=self.optimization_config['weight_decay'],
                eps=self.optimization_config['eps']
            )
        elif self.stage == 'train_caption':
            opt = torch.optim.AdamW(
                params=self.text_model.parameters(),
                lr=self.optimization_config['learning_rate'],
                betas=self.optimization_config['betas'],
                weight_decay=self.optimization_config['weight_decay'],
                eps=self.optimization_config['eps']
            )
        else:
            ValueError()

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
