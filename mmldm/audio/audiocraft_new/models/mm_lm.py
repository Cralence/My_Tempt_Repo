# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial
import logging
import math
import typing as tp
from tqdm import tqdm

import torch
from torch import nn

from ..utils import utils
from ..modules.streaming import StreamingModule, State
from ..modules.transformer import StreamingTransformer, create_norm_fn
from ..modules.conditioners import (
    ConditionFuser,
    ClassifierFreeGuidanceDropout,
    AttributeDropout,
    ConditioningProvider,
    ConditioningAttributes,
    ConditionType,
)
from ..modules.codebooks_patterns import CodebooksPatternProvider
from ..modules.activations import get_activation_fn


ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]


def get_init_fn(method: str, input_dim: int, init_depth: tp.Optional[int] = None):
    """LM layer initialization.
    Inspired from xlformers: https://github.com/fairinternal/xlformers

    Args:
        method (str): Method name for init function. Valid options are:
            'gaussian', 'uniform'.
        input_dim (int): Input dimension of the initialized module.
        init_depth (int, optional): Optional init depth value used to rescale
            the standard deviation if defined.
    """
    # Compute std
    std = 1 / math.sqrt(input_dim)
    # Rescale with depth
    if init_depth is not None:
        std = std / math.sqrt(2 * init_depth)

    if method == 'gaussian':
        return partial(
            torch.nn.init.trunc_normal_, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
    elif method == 'uniform':
        bound = math.sqrt(3) * std  # ensure the standard deviation is `std`
        return partial(torch.nn.init.uniform_, a=-bound, b=bound)
    else:
        raise ValueError("Unsupported layer initialization method")


def init_layer(m: nn.Module,
               method: str,
               init_depth: tp.Optional[int] = None,
               zero_bias_init: bool = False):
    """Wrapper around ``get_init_fn`` for proper initialization of LM modules.

    Args:
        m (nn.Module): Module to initialize.
        method (str): Method name for the init function.
        init_depth (int, optional): Optional init depth value used to rescale
            the standard deviation if defined.
        zero_bias_init (bool): Whether to initialize the bias to 0 or not.
    """
    if isinstance(m, nn.Linear):
        init_fn = get_init_fn(method, m.in_features, init_depth=init_depth)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)
        if zero_bias_init and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        init_fn = get_init_fn(method, m.embedding_dim, init_depth=None)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)


class ScaledEmbedding(nn.Embedding):
    """Boost learning rate for embeddings (with `scale`).
    """
    def __init__(self, *args, lr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def make_optim_group(self):
        group = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        return group


@dataclass
class LMOutput:
    # The logits are already re-aligned with the input codes
    # hence no extra shift is required, e.g. when computing CE
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor  # [B, K, T]


class LMModel(StreamingModule):
    """Transformer-based language model on multiple streams of codes.

    Args:
        pattern_provider (CodebooksPatternProvider): Pattern provider for codebook interleaving.
        condition_provider (MusicConditioningProvider): Conditioning provider from metadata.
        fuser (ConditionFuser): Fuser handling the fusing of conditions with language model input.
        n_q (int): Number of parallel streams to model.
        card (int): Cardinality, vocabulary size.
        dim (int): Dimension of the transformer encoder.
        num_heads (int): Number of heads for the transformer encoder.
        hidden_scale (int): Scale for hidden feed forward dimension of the transformer encoder.
        norm (str): Normalization method.
        norm_first (bool): Use pre-norm instead of post-norm.
        emb_lr (float, optional): Embedding-specific learning rate.
        bias_proj (bool): Use bias for output projections.
        weight_init (str, optional): Method for weight initialization.
        depthwise_init (str, optional): Method for depthwise weight initialization.
        zero_bias_init (bool): If true and bias in Linears, initialize bias to zeros.
        cfg_dropout (float): Classifier-free guidance dropout.
        cfg_coef (float): Classifier-free guidance coefficient.
        attribute_dropout (dict): Attribute dropout probabilities.
        two_step_cfg (bool): Whether to run classifier free-guidance with 2 distinct steps.
        **kwargs: Additional parameters for the transformer encoder.
    """
    def __init__(self, pattern_provider: CodebooksPatternProvider, condition_provider: ConditioningProvider,
                 fuser: ConditionFuser, n_q: int = 8, card: int = 1024, dim: int = 128, num_heads: int = 8,
                 hidden_scale: int = 4, norm: str = 'layer_norm', norm_first: bool = False,
                 emb_lr: tp.Optional[float] = None, bias_proj: bool = True,
                 weight_init: tp.Optional[str] = None, depthwise_init: tp.Optional[str] = None,
                 zero_bias_init: bool = False, cfg_dropout: float = 0.0, cfg_coef: float = 1.0,
                 attribute_dropout: tp.Dict[str, tp.Dict[str, float]] = {}, two_step_cfg: bool = False,
                 **kwargs):
        super().__init__()
        self.cfg_coef = cfg_coef
        self.cfg_dropout = ClassifierFreeGuidanceDropout(p=cfg_dropout)
        self.att_dropout = AttributeDropout(p=attribute_dropout)
        self.condition_provider = condition_provider
        self.fuser = fuser
        self.card = card
        embed_dim = self.card + 2  # one special token for music, and one for motion
        self.n_q = n_q
        self.dim = dim
        self.pattern_provider = pattern_provider
        self.two_step_cfg = two_step_cfg

        # zero-initialized embeddings for music and motion modality
        self.music_token_emb = nn.Parameter(torch.zeros(dim))
        self.motion_token_emb = nn.Parameter(torch.zeros(dim))

        self.emb = nn.ModuleList([ScaledEmbedding(embed_dim, dim, lr=emb_lr) for _ in range(n_q)])

        if 'activation' in kwargs:
            kwargs['activation'] = get_activation_fn(kwargs['activation'])
        self.transformer = StreamingTransformer(
            d_model=dim, num_heads=num_heads, dim_feedforward=int(hidden_scale * dim),
            norm=norm, norm_first=norm_first, **kwargs)

        self.out_norm: tp.Optional[nn.Module] = None
        if norm_first:
            self.out_norm = create_norm_fn(norm, dim)

        self.linears = nn.ModuleList([nn.Linear(dim, self.card, bias=bias_proj) for _ in range(n_q)])
        self.motion_linears = nn.ModuleList([nn.Linear(dim, self.card, bias=bias_proj) for _ in range(n_q)])

        self._init_weights(weight_init, depthwise_init, zero_bias_init)
        self._fsdp: tp.Optional[nn.Module]
        self.__dict__['_fsdp'] = None

    def _init_weights(self, weight_init: tp.Optional[str], depthwise_init: tp.Optional[str], zero_bias_init: bool):
        """Initialization of the transformer module weights.

        Args:
            weight_init (str, optional): Weight initialization strategy. See ``get_init_fn`` for valid options.
            depthwise_init (str, optional): Depthwise initialization strategy. The following options are valid:
                'current' where the depth corresponds to the current layer index or 'global' where the total number
                of layer is used as depth. If not set, no depthwise initialization strategy is used.
            zero_bias_init (bool): Whether to initialize bias to zero or not.
        """
        assert depthwise_init is None or depthwise_init in ['current', 'global']
        assert depthwise_init is None or weight_init is not None, \
            "If 'depthwise_init' is defined, a 'weight_init' method should be provided."
        assert not zero_bias_init or weight_init is not None, \
            "If 'zero_bias_init', a 'weight_init' method should be provided"

        if weight_init is None:
            return

        for emb_layer in self.emb:
            init_layer(emb_layer, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

        for layer_idx, tr_layer in enumerate(self.transformer.layers):
            depth = None
            if depthwise_init == 'current':
                depth = layer_idx + 1
            elif depthwise_init == 'global':
                depth = len(self.transformer.layers)
            init_fn = partial(init_layer, method=weight_init, init_depth=depth, zero_bias_init=zero_bias_init)
            tr_layer.apply(init_fn)

        for linear in self.linears:
            init_layer(linear, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

        for linear in self.motion_linears:
            init_layer(linear, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

    @property
    def music_special_token_id(self) -> int:
        return self.card

    @property
    def motion_special_token_id(self) -> int:
        return self.card + 1

    @property
    def num_codebooks(self) -> int:
        return self.n_q

    def forward(self,
                sequence: torch.Tensor,
                conditions: tp.List[ConditioningAttributes],
                src_mask: tp.Optional[torch.Tensor] = None,
                condition_tensors: tp.Optional[ConditionTensors] = None,
                return_last_layer: bool = False) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor] :
        """Apply language model on sequence and conditions.
        Given a tensor of sequence of shape [B, K, S] with K the number of codebooks and
        S the sequence steps, return the logits with shape [B, card, K, S].

        Args:
            indices (torch.Tensor): Indices of the codes to model.
            conditions (list of ConditioningAttributes): Conditions to use when modeling
                the given codes. Note that when evaluating multiple time with the same conditioning
                you should pre-compute those and pass them as `condition_tensors`.
            condition_tensors (dict[str, ConditionType], optional): Pre-computed conditioning
                tensors, see `conditions`.
        Returns:
            torch.Tensor: Logits.
        """
        B, K, S = sequence.shape
        assert K == self.num_codebooks, "Sequence shape must match the specified number of codebooks"

        input_ = sum([self.emb[k](sequence[:, k]) for k in range(K)])  # b, s, dim

        if condition_tensors is None:
            assert not self._is_streaming, "Conditions tensors should be precomputed when streaming."
            # apply dropout modules
            conditions = self.cfg_dropout(conditions)
            conditions = self.att_dropout(conditions)
            tokenized = self.condition_provider.tokenize(conditions)
            # encode conditions and fuse, both have a streaming cache to not recompute when generating.
            condition_tensors = self.condition_provider(tokenized)
        else:
            assert not conditions, "Shouldn't pass both conditions and condition_tensors."

        # add music and motion embedding
        input_[:, :S // 2] += self.music_token_emb
        input_[:, S // 2:] += self.motion_token_emb

        input_, cross_attention_input = self.fuser(input_, condition_tensors)

        out = self.transformer(input_, separate_positional_encoding=True, cross_attention_src=cross_attention_input, src_mask=src_mask)
        if self.out_norm:
            out = self.out_norm(out)

        if return_last_layer:
            return out  # [B, S, D]

        music_logits = torch.stack([self.linears[k](out[:, :S // 2]) for k in range(K)], dim=1)  # [B, K, S/2, card]
        motion_logits = torch.stack([self.motion_linears[k](out[:, S // 2:]) for k in range(K)], dim=1)   # [B, K, S/2, card]

        # remove the prefix from the model outputs
        if len(self.fuser.fuse2cond['prepend']) > 0:
            music_logits = music_logits[:, :, -S:]
            motion_logits = motion_logits[:, :, -S:]

        return music_logits, motion_logits  # [B, K, S, card]

    def compute_predictions(
            self, music_codes: torch.Tensor,
            motion_codes: torch.Tensor,
            conditions: tp.List[ConditioningAttributes],
            condition_tensors: tp.Optional[ConditionTensors] = None) -> tp.Tuple[LMOutput, LMOutput]:
        """Given an input tensor of codes [B, K, T] and list of conditions, runs the model
        forward using the specified codes interleaving pattern.

        Args:
            codes (torch.Tensor): Input codes of shape [B, K, T] with B the batch size,
                K the number of codebooks and T the number of timesteps.
            conditions (list of ConditioningAttributes): conditionings to use when modeling
                the given codes. Note that when evaluating multiple time with the same conditioning
                you should pre-compute those and pass them as `condition_tensors`.
            condition_tensors (dict[str, ConditionType], optional): pre-computed conditioning
                tensors, see `conditions`.
        Returns:
            LMOutput: Language model outputs
                logits (torch.Tensor) of shape [B, K, T, card] corresponding to the provided codes,
                    i.e. the first item corresponds to logits to predict the first code, meaning that
                    no additional shifting of codes and logits is required.
                mask (torch.Tensor) of shape [B, K, T], mask over valid and invalid positions.
                    Given the specified interleaving strategies, parts of the logits and codes should
                    not be considered as valid predictions because of invalid context.
        """
        # prepare input sequence
        B, K, T_music = music_codes.shape
        T_motion = motion_codes.shape[-1]
        assert T_music == T_motion
        music_codes = music_codes.contiguous()
        motion_codes = motion_codes.contiguous()
        # map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
        music_pattern = self.pattern_provider.get_pattern(T_music)
        motion_pattern = self.pattern_provider.get_pattern(T_motion)
        music_sequence_codes, _, _ = music_pattern.build_pattern_sequence(
            music_codes, self.music_special_token_id, keep_only_valid_steps=True
        )
        motion_sequence_codes, _, _ = motion_pattern.build_pattern_sequence(
            motion_codes, self.motion_special_token_id, keep_only_valid_steps=True
        )
        sequence_codes = torch.cat((music_sequence_codes, motion_sequence_codes), dim=-1)

        # prepare self-attention mask
        self_attn_map = self.get_self_attn_mask(music_sequence_codes.shape[-1], motion_sequence_codes.shape[-1])

        # apply model on pattern sequence
        music_logits, motion_logits = self(sequence_codes, conditions, src_mask=self_attn_map, condition_tensors=condition_tensors)  # [B, K, S, card]
        # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
        # and provide the corresponding mask over invalid positions of tokens
        music_logits = music_logits.permute(0, 3, 1, 2)  # [B, card, K, S]
        motion_logits = motion_logits.permute(0, 3, 1, 2)
        # note: we use nans as special token to make it obvious if we feed unexpected logits
        music_logits, _, music_logits_mask = music_pattern.revert_pattern_logits(
            music_logits, float('nan'), keep_only_valid_steps=True
        )
        motion_logits, _, motion_logits_mask = motion_pattern.revert_pattern_logits(
            motion_logits, float('nan'), keep_only_valid_steps=True
        )
        music_logits = music_logits.permute(0, 2, 3, 1)  # [B, K, T, card]
        music_logits_mask = music_logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]
        motion_logits = motion_logits.permute(0, 2, 3, 1)  # [B, K, T, card]
        motion_logits_mask = motion_logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]

        return LMOutput(music_logits, music_logits_mask), LMOutput(motion_logits, motion_logits_mask)

    def get_music_motion_context(
            self, music_codes: torch.Tensor,
            motion_codes: torch.Tensor,
            conditions: tp.List[ConditioningAttributes],
            condition_tensors: tp.Optional[ConditionTensors] = None) -> torch.Tensor:
        # prepare input sequence
        B, K, T_music = music_codes.shape
        T_motion = motion_codes.shape[-1]
        assert T_music == T_motion
        music_codes = music_codes.contiguous()
        motion_codes = motion_codes.contiguous()
        # map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
        music_pattern = self.pattern_provider.get_pattern(T_music)
        motion_pattern = self.pattern_provider.get_pattern(T_motion)
        music_sequence_codes, _, _ = music_pattern.build_pattern_sequence(
            music_codes, self.music_special_token_id, keep_only_valid_steps=True
        )
        motion_sequence_codes, _, _ = motion_pattern.build_pattern_sequence(
            motion_codes, self.motion_special_token_id, keep_only_valid_steps=True
        )
        sequence_codes = torch.cat((music_sequence_codes, motion_sequence_codes), dim=-1)

        # prepare self-attention mask
        self_attn_map = self.get_self_attn_mask(music_sequence_codes.shape[-1], motion_sequence_codes.shape[-1])

        # apply model on pattern sequence
        music_motion_context = self(sequence_codes, conditions, src_mask=self_attn_map,
                                    condition_tensors=condition_tensors, return_last_layer=True)  # [B, S, D]

        return music_motion_context

    def get_self_attn_mask(self, section_1: int, section_2: int) -> torch.Tensor:
        device = next(iter(self.parameters())).device
        mask = torch.zeros((section_1 + section_2, section_1 + section_2), dtype=torch.bool, device=device)

        mask[:section_1, :section_1] = ~torch.ones((section_1, section_1), dtype=torch.bool, device=device).triu(1)
        mask[section_1:section_1 + section_2, :section_2] = ~torch.ones((section_2, section_2), dtype=torch.bool, device=device).triu(1)
        mask[:section_2, section_1:section_1 + section_2] = ~torch.ones((section_2, section_2), dtype=torch.bool, device=device).triu(1)
        mask[section_1:section_1 + section_2, section_1:section_1 + section_2] = ~torch.ones((section_2, section_2), dtype=torch.bool, device=device).triu(1)

        mask = torch.where(mask, 0., float('-inf'))
        return mask

    def _sample_next_token(self,
                           music_sequence: torch.Tensor,
                           motion_sequence: torch.Tensor,
                           cfg_conditions: CFGConditions,
                           unconditional_state: State,
                           use_sampling: bool = False,
                           temp: float = 1.0,
                           top_k: int = 0,
                           top_p: float = 0.0,
                           cfg_coef: tp.Optional[float] = None,
                           two_step_cfg: tp.Optional[bool] = None) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Sample next token from the model given a sequence and a set of conditions. The model supports
        multiple sampling strategies (greedy sampling, softmax, top-k, top-p...).

        Args:
            sequence (torch.Tensor): Current sequence of shape [B, K, S]
                with K corresponding to the number of codebooks and S the number of sequence steps.
                S = 1 in streaming mode, except for the first step that contains a bigger prompt.
            condition_tensors (dict[str, ConditionType): Set of conditions. If CFG is used,
                should be twice the batch size, being the concatenation of the conditions + null conditions.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            cfg_coef (float, optional): classifier free guidance coefficient
        Returns:
            next_token (torch.Tensor): Next token tensor of shape [B, K, 1].
        """
        B = music_sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef
        two_step_cfg = self.two_step_cfg if two_step_cfg is None else two_step_cfg

        sequence = torch.cat((music_sequence, motion_sequence), dim=-1)
        src_mask = self.get_self_attn_mask(music_sequence.shape[-1], motion_sequence.shape[-1])

        if two_step_cfg and cfg_conditions != {}:
            assert isinstance(cfg_conditions, tuple), type(cfg_conditions)
            condition_tensors, null_condition_tensors = cfg_conditions
            music_cond_logits, motion_cond_logits = self(sequence, conditions=[], condition_tensors=condition_tensors, src_mask=src_mask)
            state = self.get_streaming_state()
            self.set_streaming_state(unconditional_state)
            music_uncond_logits, motion_uncond_logits = self(sequence, conditions=[], condition_tensors=null_condition_tensors, src_mask=src_mask)
            unconditional_state.update(self.get_streaming_state())
            self.set_streaming_state(state)
            music_logits = music_uncond_logits + (music_cond_logits - music_uncond_logits) * self.cfg_coef
            motion_logits = motion_uncond_logits + (motion_cond_logits - motion_uncond_logits) * self.cfg_coef
        else:
            assert isinstance(cfg_conditions, dict)
            condition_tensors = cfg_conditions
            if condition_tensors:
                # Preparing for CFG, predicting both conditional and unconditional logits.
                sequence = torch.cat([sequence, sequence], dim=0)
            music_all_logits, motion_all_logits = self(
                sequence,
                conditions=[], condition_tensors=condition_tensors, src_mask=src_mask)
            if condition_tensors:
                music_cond_logits, music_uncond_logits = music_all_logits.split(B, dim=0)  # [B, K, T, card]
                motion_cond_logits, motion_uncond_logits = motion_all_logits.split(B, dim=0)  # [B, K, T, card]
                music_logits = music_uncond_logits + (music_cond_logits - music_uncond_logits) * cfg_coef
                motion_logits = motion_uncond_logits + (motion_cond_logits - motion_uncond_logits) * cfg_coef
            else:
                music_logits = music_all_logits
                motion_logits = motion_all_logits

        # sample music tokne
        music_logits = music_logits.permute(0, 1, 3, 2)  # [B, K, card, T]
        music_logits = music_logits[..., -1]  # [B x K x card]

        # Apply softmax for sampling if temp > 0. Else, do greedy sampling to avoid zero division error.
        if use_sampling and temp > 0.0:
            probs = torch.softmax(music_logits / temp, dim=-1)
            if top_p > 0.0:
                music_next_token = utils.sample_top_p(probs, p=top_p)
            elif top_k > 0:
                music_next_token = utils.sample_top_k(probs, k=top_k)
            else:
                music_next_token = utils.multinomial(probs, num_samples=1)
        else:
            music_next_token = torch.argmax(music_logits, dim=-1, keepdim=True)

        # sample music tokne
        motion_logits = motion_logits.permute(0, 1, 3, 2)  # [B, K, card, T]
        motion_logits = motion_logits[..., -1]  # [B x K x card]

        # Apply softmax for sampling if temp > 0. Else, do greedy sampling to avoid zero division error.
        if use_sampling and temp > 0.0:
            probs = torch.softmax(motion_logits / temp, dim=-1)
            if top_p > 0.0:
                motion_next_token = utils.sample_top_p(probs, p=top_p)
            elif top_k > 0:
                motion_next_token = utils.sample_top_k(probs, k=top_k)
            else:
                motion_next_token = utils.multinomial(probs, num_samples=1)
        else:
            motion_next_token = torch.argmax(motion_logits, dim=-1, keepdim=True)

        return music_next_token, motion_next_token

    @torch.no_grad()
    def generate(
            self,
            conditions: tp.List[ConditioningAttributes] = [],
            music_code: tp.Optional[torch.LongTensor] = None,
            motion_code: tp.Optional[torch.LongTensor] = None,
            num_samples: tp.Optional[int] = None,
            max_gen_len: int = 256,
            use_sampling: bool = True,
            temp: float = 1.0,
            top_k: int = 250,
            top_p: float = 0.0,
            cfg_coef: tp.Optional[float] = None,
            two_step_cfg: tp.Optional[bool] = None,
            remove_prompts: bool = False,
            check: bool = True,
            callback: tp.Optional[tp.Callable[[int, int], None]] = None
    ) -> tp.Tuple[torch.LongTensor, torch.LongTensor]:
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device

        assert music_code is None or motion_code is None, "cannot provide both music and motion code."

        # Checking all input shapes are consistent.
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif conditions:
            possible_num_samples.append(len(conditions))
        else:
            possible_num_samples.append(1)
        assert [x == possible_num_samples[0] for x in possible_num_samples], "Inconsistent inputs shapes"
        num_samples = possible_num_samples[0]

        # below we create set of conditions: one conditional and one unconditional
        # to do that we merge the regular condition together with the null condition
        # we then do 1 forward pass instead of 2.
        # the reason for that is two-fold:
        # 1. it is about x2 faster than doing 2 forward passes
        # 2. avoid the streaming API treating the 2 passes as part of different time steps
        # We also support doing two different passes, in particular to ensure that
        # the padding structure is exactly the same between train and test.
        # With a batch size of 1, this can be slower though.
        cfg_conditions: CFGConditions
        two_step_cfg = self.two_step_cfg if two_step_cfg is None else two_step_cfg
        if conditions:
            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
            if two_step_cfg:
                cfg_conditions = (
                    self.condition_provider(self.condition_provider.tokenize(conditions, device)),
                    self.condition_provider(self.condition_provider.tokenize(null_conditions, device)),
                )
            else:
                conditions = conditions + null_conditions
                tokenized = self.condition_provider.tokenize(conditions, device)
                cfg_conditions = self.condition_provider(tokenized)
        else:
            cfg_conditions = {}

        assert num_samples > 0
        prompt = torch.zeros((num_samples, self.num_codebooks, 0), dtype=torch.long, device=device)

        B, K, T = prompt.shape
        start_offset = T
        assert start_offset < max_gen_len

        pattern = self.pattern_provider.get_pattern(max_gen_len)
        # this token is used as default value for codes that are not generated yet
        unknown_token = -1

        # we generate codes up to the max_gen_len that will be mapped to the pattern sequence
        # replace the unknown code with provided code if necessary
        if music_code is None:
            music_gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)
        else:
            music_gen_codes = music_code
        if motion_code is None:
            motion_gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)
        else:
            motion_gen_codes = motion_code
        assert music_gen_codes.shape[-1] == motion_gen_codes.shape[-1], "music code and motion code should be in equal time dimension"
        # filling the gen_codes with the prompt if needed
        music_gen_codes[..., :start_offset] = prompt
        motion_gen_codes[..., :start_offset] = prompt
        # create the gen_sequence with proper interleaving from the pattern: [B, K, S]
        music_gen_sequence, _, music_mask = pattern.build_pattern_sequence(music_gen_codes, self.music_special_token_id)   # gen_sequence: padded with self.music_special_token_id
        motion_gen_sequence, _, motion_mask = pattern.build_pattern_sequence(motion_gen_codes, self.motion_special_token_id)   # gen_sequence: padded with self.motion_special_token_id
        # retrieve the start_offset in the sequence:
        # it is the first sequence step that contains the `start_offset` timestep
        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)
        assert start_offset_sequence is not None

        unconditional_state = self.get_streaming_state()
        prev_offset = 0
        gen_sequence_len = music_gen_sequence.shape[-1]  # gen_sequence shape is [B, K, S]
        for offset in tqdm(range(start_offset_sequence, gen_sequence_len), desc="Generating music & motion"):
            # get current sequence (note that the streaming API is providing the caching over previous offsets)
            music_curr_sequence = music_gen_sequence[..., prev_offset:offset]
            music_curr_mask = music_mask[None, ..., prev_offset:offset].expand(B, -1, -1)
            motion_curr_sequence = motion_gen_sequence[..., prev_offset:offset]
            motion_curr_mask = motion_mask[None, ..., prev_offset:offset].expand(B, -1, -1)
            if check:
                # check coherence between mask and sequence
                assert (music_curr_sequence == torch.where(music_curr_mask, music_curr_sequence, self.music_special_token_id)).all()
                assert (motion_curr_sequence == torch.where(motion_curr_mask, motion_curr_sequence, self.motion_special_token_id)).all()
                # should never happen as gen_sequence is filled progressively
                assert not (music_curr_sequence == unknown_token).any()
                assert not (motion_curr_sequence == unknown_token).any()
            # sample next token from the model, next token shape is [B, K, 1]
            music_next_token, motion_next_token = self._sample_next_token(
                music_curr_sequence, motion_curr_sequence, cfg_conditions, unconditional_state, use_sampling,
                temp, top_k, top_p, cfg_coef=cfg_coef, two_step_cfg=two_step_cfg)
            # ensure the tokens that should be masked are properly set to special_token_id
            # as the model never output special_token_id
            music_valid_mask = music_mask[..., offset:offset+1].expand(B, -1, -1)
            music_next_token[~music_valid_mask] = self.music_special_token_id
            motion_valid_mask = motion_mask[..., offset:offset + 1].expand(B, -1, -1)
            motion_next_token[~motion_valid_mask] = self.motion_special_token_id
            # ensure we don't overwrite prompt tokens, we only write over unknown tokens
            # (then mask tokens should be left as is as well, which is correct)
            # then update the prediction if they are not provided
            if music_code is None:
                music_gen_sequence[..., offset:offset+1] = torch.where(
                    music_gen_sequence[..., offset:offset+1] == unknown_token,
                    music_next_token, music_gen_sequence[..., offset:offset+1]
                )
            if motion_code is None:
                motion_gen_sequence[..., offset:offset + 1] = torch.where(
                    motion_gen_sequence[..., offset:offset + 1] == unknown_token,
                    motion_next_token, motion_gen_sequence[..., offset:offset + 1]
                )
            if callback is not None:
                callback(1 + offset - start_offset_sequence, gen_sequence_len - start_offset_sequence)
        unconditional_state.clear()

        # ensure sequence has been entirely filled
        assert not (music_gen_sequence == unknown_token).any()
        assert not (motion_gen_sequence == unknown_token).any()
        # ensure gen_sequence pattern and mask are matching
        # which means the gen_sequence is valid according to the pattern
        assert (
            music_gen_sequence == torch.where(music_mask[None, ...].expand(B, -1, -1), music_gen_sequence,
                                              self.music_special_token_id)).all()
        assert (
            motion_gen_sequence == torch.where(motion_mask[None, ...].expand(B, -1, -1), motion_gen_sequence,
                                               self.motion_special_token_id)).all()
        # get back the codes, trimming the prompt if needed and cutting potentially incomplete timesteps
        music_out_codes, out_indexes, music_out_mask = pattern.revert_pattern_sequence(music_gen_sequence, special_token=unknown_token)
        motion_out_codes, out_indexes, motion_out_mask = pattern.revert_pattern_sequence(motion_gen_sequence, special_token=unknown_token)

        # sanity checks over the returned codes and corresponding masks
        assert (music_out_codes[..., :max_gen_len] != unknown_token).all()
        assert (music_out_mask[..., :max_gen_len] == 1).all()
        assert (motion_out_codes[..., :max_gen_len] != unknown_token).all()
        assert (motion_out_mask[..., :max_gen_len] == 1).all()

        out_start_offset = start_offset if remove_prompts else 0
        music_out_codes = music_out_codes[..., out_start_offset:max_gen_len]
        motion_out_codes = motion_out_codes[..., out_start_offset:max_gen_len]

        # ensure the returned codes are all valid
        assert (music_out_codes >= 0).all() and (music_out_codes <= self.card).all()
        assert (motion_out_codes >= 0).all() and (motion_out_codes <= self.card).all()
        return music_out_codes, motion_out_codes
