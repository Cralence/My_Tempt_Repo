import io
import math
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
import importlib.util

import torch
from torch import Tensor


def is_module_available(*modules: str) -> bool:
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    return all(importlib.util.find_spec(m) is not None for m in modules)


_IS_TORCHAUDIO_EXT_AVAILABLE = False  # is_module_available("torchaudio.lib._torchaudio")


def spectrogram(
    waveform: Tensor,
    pad: int,
    window: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    power: Optional[float],
    normalized: Union[bool, str],
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    return_complex: Optional[bool] = None,
) -> Tensor:
    r"""Create a spectrogram or a batch of spectrograms from a raw audio signal.
    The spectrogram can be either magnitude-only or complex.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (Tensor): Tensor of audio of dimension `(..., time)`
        pad (int): Two sided padding of signal
        window (Tensor): Window tensor that is applied/multiplied to each frame/window
        n_fft (int): Size of FFT
        hop_length (int): Length of hop between STFT windows
        win_length (int): Window size
        power (float or None): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc.
            If None, then the complex spectrum is returned instead.
        normalized (bool or str): Whether to normalize by magnitude after stft. If input is str, choices are
            ``"window"`` and ``"frame_length"``, if specific normalization type is desirable. ``True`` maps to
            ``"window"``. When normalized on ``"window"``, waveform is normalized upon the window's L2 energy. If
            normalized on ``"frame_length"``, waveform is normalized by dividing by
            :math:`(\text{frame\_length})^{0.5}`.
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            Default: ``True``
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. Default: ``"reflect"``
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy. Default: ``True``
        return_complex (bool, optional):
            Deprecated and not used.

    Returns:
        Tensor: Dimension `(..., freq, time)`, freq is
        ``n_fft // 2 + 1`` and ``n_fft`` is the number of
        Fourier bins, and time is the number of window hops (n_frame).
    """
    if return_complex is not None:
        warnings.warn(
            "`return_complex` argument is now deprecated and is not effective."
            "`torchaudio.functional.spectrogram(power=None)` always returns a tensor with "
            "complex dtype. Please remove the argument in the function call."
        )

    if pad > 0:
        # TODO add "with torch.no_grad():" back when JIT supports it
        waveform = torch.nn.functional.pad(waveform, (pad, pad), "constant")

    frame_length_norm, window_norm = _get_spec_norms(normalized)

    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    # default values are consistent with librosa.core.spectrum._spectrogram
    spec_f = torch.stft(
        input=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=frame_length_norm,
        onesided=onesided,
        return_complex=True,
    )

    # unpack batch
    spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-2:])

    if window_norm:
        spec_f /= window.pow(2.0).sum().sqrt()
    if power is not None:
        if power == 1.0:
            return spec_f.abs()
        return spec_f.abs().pow(power)
    return spec_f


def _get_spec_norms(normalized: Union[str, bool]):
    frame_length_norm, window_norm = False, False
    if torch.jit.isinstance(normalized, str):
        if normalized not in ["frame_length", "window"]:
            raise ValueError("Invalid normalized parameter: {}".format(normalized))
        if normalized == "frame_length":
            frame_length_norm = True
        elif normalized == "window":
            window_norm = True
    elif torch.jit.isinstance(normalized, bool):
        if normalized:
            window_norm = True
    else:
        raise TypeError("Input type not supported")
    return frame_length_norm, window_norm


def melscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
) -> Tensor:
    r"""Create a frequency bin conversion matrix.

    .. devices:: CPU

    .. properties:: TorchScript

    Note:
        For the sake of the numerical compatibility with librosa, not all the coefficients
        in the resulting filter bank has magnitude of 1.

        .. image:: https://download.pytorch.org/torchaudio/doc-assets/mel_fbanks.png
           :alt: Visualization of generated filter bank

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_mels (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform
        norm (str or None, optional): If "slaney", divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_mels``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A * melscale_fbanks(A.size(-1), ...)``.

    """

    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')

    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    m_min = _hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale=mel_scale)

    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)

    if (fb.max(dim=0).values == 0.0).any():
        warnings.warn(
            "At least one mel filterbank has all zero values. "
            f"The value for `n_mels` ({n_mels}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )

    return fb


def _hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    r"""Convert Hz to Mels.

    Args:
        freqs (float): Frequencies in Hz
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        mels (float): Frequency in Mels
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq / 700.0))

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (freq - f_min) / f_sp

    # Fill in the log-scale part
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep

    return mels


def _mel_to_hz(mels: Tensor, mel_scale: str = "htk") -> Tensor:
    """Convert mel bin numbers to frequencies.

    Args:
        mels (Tensor): Mel frequencies
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        freqs (Tensor): Mels converted in Hz
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))

    return freqs


def _create_triangular_filterbank(
    all_freqs: Tensor,
    f_pts: Tensor,
) -> Tensor:
    """Create a triangular filter bank.

    Args:
        all_freqs (Tensor): STFT freq points of size (`n_freqs`).
        f_pts (Tensor): Filter mid points of size (`n_filter`).

    Returns:
        fb (Tensor): The filter bank of size (`n_freqs`, `n_filter`).
    """
    # Adopted from Librosa
    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    return fb


def loudness(waveform: Tensor, sample_rate: int):
    r"""Measure audio loudness according to the ITU-R BS.1770-4 recommendation.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    Args:
        waveform(torch.Tensor): audio waveform of dimension `(..., channels, time)`
        sample_rate (int): sampling rate of the waveform

    Returns:
        Tensor: loudness estimates (LKFS)

    Reference:
        - https://www.itu.int/rec/R-REC-BS.1770-4-201510-I/en
    """

    if waveform.size(-2) > 5:
        raise ValueError("Only up to 5 channels are supported.")

    gate_duration = 0.4
    overlap = 0.75
    gamma_abs = -70.0
    kweight_bias = -0.691
    gate_samples = int(round(gate_duration * sample_rate))
    step = int(round(gate_samples * (1 - overlap)))

    # Apply K-weighting
    waveform = treble_biquad(waveform, sample_rate, 4.0, 1500.0, 1 / math.sqrt(2))
    waveform = highpass_biquad(waveform, sample_rate, 38.0, 0.5)

    # Compute the energy for each block
    energy = torch.square(waveform).unfold(-1, gate_samples, step)
    energy = torch.mean(energy, dim=-1)

    # Compute channel-weighted summation
    g = torch.tensor([1.0, 1.0, 1.0, 1.41, 1.41], dtype=waveform.dtype, device=waveform.device)
    g = g[: energy.size(-2)]

    energy_weighted = torch.sum(g.unsqueeze(-1) * energy, dim=-2)
    loudness = -0.691 + 10 * torch.log10(energy_weighted)

    # Apply absolute gating of the blocks
    gated_blocks = loudness > gamma_abs
    gated_blocks = gated_blocks.unsqueeze(-2)

    energy_filtered = torch.sum(gated_blocks * energy, dim=-1) / torch.count_nonzero(gated_blocks, dim=-1)
    energy_weighted = torch.sum(g * energy_filtered, dim=-1)
    gamma_rel = kweight_bias + 10 * torch.log10(energy_weighted) - 10

    # Apply relative gating of the blocks
    gated_blocks = torch.logical_and(gated_blocks.squeeze(-2), loudness > gamma_rel.unsqueeze(-1))
    gated_blocks = gated_blocks.unsqueeze(-2)

    energy_filtered = torch.sum(gated_blocks * energy, dim=-1) / torch.count_nonzero(gated_blocks, dim=-1)
    energy_weighted = torch.sum(g * energy_filtered, dim=-1)
    LKFS = kweight_bias + 10 * torch.log10(energy_weighted)
    return LKFS


def highpass_biquad(waveform: Tensor, sample_rate: int, cutoff_freq: float, Q: float = 0.707) -> Tensor:
    r"""Design biquad highpass filter and perform filtering.  Similar to SoX implementation.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        cutoff_freq (float or torch.Tensor): filter cutoff frequency
        Q (float or torch.Tensor, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``)

    Returns:
        Tensor: Waveform dimension of `(..., time)`
    """
    dtype = waveform.dtype
    device = waveform.device
    cutoff_freq = torch.as_tensor(cutoff_freq, dtype=dtype, device=device)
    Q = torch.as_tensor(Q, dtype=dtype, device=device)

    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = torch.sin(w0) / 2.0 / Q

    b0 = (1 + torch.cos(w0)) / 2
    b1 = -1 - torch.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * torch.cos(w0)
    a2 = 1 - alpha
    return biquad(waveform, b0, b1, b2, a0, a1, a2)


def treble_biquad(
    waveform: Tensor,
    sample_rate: int,
    gain: float,
    central_freq: float = 3000,
    Q: float = 0.707,
) -> Tensor:
    r"""Design a treble tone-control effect.  Similar to SoX implementation.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        gain (float or torch.Tensor): desired gain at the boost (or attenuation) in dB.
        central_freq (float or torch.Tensor, optional): central frequency (in Hz). (Default: ``3000``)
        Q (float or torch.Tensor, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``).

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    Reference:
        - http://sox.sourceforge.net/sox.html
        - https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF
    """
    dtype = waveform.dtype
    device = waveform.device
    central_freq = torch.as_tensor(central_freq, dtype=dtype, device=device)
    Q = torch.as_tensor(Q, dtype=dtype, device=device)
    gain = torch.as_tensor(gain, dtype=dtype, device=device)

    w0 = 2 * math.pi * central_freq / sample_rate
    alpha = torch.sin(w0) / 2 / Q
    A = torch.exp(gain / 40 * math.log(10))

    temp1 = 2 * torch.sqrt(A) * alpha
    temp2 = (A - 1) * torch.cos(w0)
    temp3 = (A + 1) * torch.cos(w0)

    b0 = A * ((A + 1) + temp2 + temp1)
    b1 = -2 * A * ((A - 1) + temp3)
    b2 = A * ((A + 1) + temp2 - temp1)
    a0 = (A + 1) - temp2 + temp1
    a1 = 2 * ((A - 1) - temp3)
    a2 = (A + 1) - temp2 - temp1

    return biquad(waveform, b0, b1, b2, a0, a1, a2)


def biquad(waveform: Tensor, b0: float, b1: float, b2: float, a0: float, a1: float, a2: float) -> Tensor:
    r"""Perform a biquad filter of input tensor.  Initial conditions set to 0.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        b0 (float or torch.Tensor): numerator coefficient of current input, x[n]
        b1 (float or torch.Tensor): numerator coefficient of input one time step ago x[n-1]
        b2 (float or torch.Tensor): numerator coefficient of input two time steps ago x[n-2]
        a0 (float or torch.Tensor): denominator coefficient of current output y[n], typically 1
        a1 (float or torch.Tensor): denominator coefficient of current output y[n-1]
        a2 (float or torch.Tensor): denominator coefficient of current output y[n-2]

    Returns:
        Tensor: Waveform with dimension of `(..., time)`

    Reference:
       - https://en.wikipedia.org/wiki/Digital_biquad_filter
    """

    device = waveform.device
    dtype = waveform.dtype

    b0 = torch.as_tensor(b0, dtype=dtype, device=device).view(1)
    b1 = torch.as_tensor(b1, dtype=dtype, device=device).view(1)
    b2 = torch.as_tensor(b2, dtype=dtype, device=device).view(1)
    a0 = torch.as_tensor(a0, dtype=dtype, device=device).view(1)
    a1 = torch.as_tensor(a1, dtype=dtype, device=device).view(1)
    a2 = torch.as_tensor(a2, dtype=dtype, device=device).view(1)

    output_waveform = lfilter(
        waveform,
        torch.cat([a0, a1, a2]),
        torch.cat([b0, b1, b2]),
    )
    return output_waveform


def _lfilter_core_generic_loop(input_signal_windows: Tensor, a_coeffs_flipped: Tensor, padded_output_waveform: Tensor):
    n_order = a_coeffs_flipped.size(1)
    a_coeffs_flipped = a_coeffs_flipped.unsqueeze(2)
    for i_sample, o0 in enumerate(input_signal_windows.permute(2, 0, 1)):
        windowed_output_signal = padded_output_waveform[:, :, i_sample : i_sample + n_order]
        o0 -= (windowed_output_signal.transpose(0, 1) @ a_coeffs_flipped)[..., 0].t()
        padded_output_waveform[:, :, i_sample + n_order - 1] = o0


if _IS_TORCHAUDIO_EXT_AVAILABLE:
    _lfilter_core_cpu_loop = torch.ops.torchaudio._lfilter_core_loop
else:
    _lfilter_core_cpu_loop = _lfilter_core_generic_loop


def _lfilter_core(
    waveform: Tensor,
    a_coeffs: Tensor,
    b_coeffs: Tensor,
) -> Tensor:

    if a_coeffs.size() != b_coeffs.size():
        raise ValueError(
            "Expected coeffs to be the same size."
            f"Found a_coeffs size: {a_coeffs.size()}, b_coeffs size: {b_coeffs.size()}"
        )
    if waveform.ndim != 3:
        raise ValueError(f"Expected waveform to be 3 dimensional. Found: {waveform.ndim}")
    if not (waveform.device == a_coeffs.device == b_coeffs.device):
        raise ValueError(
            "Expected waveform and coeffs to be on the same device."
            f"Found: waveform device:{waveform.device}, a_coeffs device: {a_coeffs.device}, "
            f"b_coeffs device: {b_coeffs.device}"
        )

    n_batch, n_channel, n_sample = waveform.size()
    n_order = a_coeffs.size(1)
    if n_order <= 0:
        raise ValueError(f"Expected n_order to be positive. Found: {n_order}")

    # Pad the input and create output

    padded_waveform = torch.nn.functional.pad(waveform, [n_order - 1, 0])
    padded_output_waveform = torch.zeros_like(padded_waveform)

    # Set up the coefficients matrix
    # Flip coefficients' order
    a_coeffs_flipped = a_coeffs.flip(1)
    b_coeffs_flipped = b_coeffs.flip(1)

    # calculate windowed_input_signal in parallel using convolution
    input_signal_windows = torch.nn.functional.conv1d(padded_waveform, b_coeffs_flipped.unsqueeze(1), groups=n_channel)

    input_signal_windows.div_(a_coeffs[:, :1])
    a_coeffs_flipped.div_(a_coeffs[:, :1])

    if (
        input_signal_windows.device == torch.device("cpu")
        and a_coeffs_flipped.device == torch.device("cpu")
        and padded_output_waveform.device == torch.device("cpu")
    ):
        _lfilter_core_cpu_loop(input_signal_windows, a_coeffs_flipped, padded_output_waveform)
    else:
        _lfilter_core_generic_loop(input_signal_windows, a_coeffs_flipped, padded_output_waveform)

    output = padded_output_waveform[:, :, n_order - 1 :]
    return output


if _IS_TORCHAUDIO_EXT_AVAILABLE:
    _lfilter = torch.ops.torchaudio._lfilter
else:
    _lfilter = _lfilter_core


def lfilter(waveform: Tensor, a_coeffs: Tensor, b_coeffs: Tensor, clamp: bool = True, batching: bool = True) -> Tensor:
    r"""Perform an IIR filter by evaluating difference equation.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Note:
        To avoid numerical problems, small filter order is preferred.
        Using double precision could also minimize numerical precision errors.

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`.  Must be normalized to -1 to 1.
        a_coeffs (Tensor): denominator coefficients of difference equation of dimension of either
                                1D with shape `(num_order + 1)` or 2D with shape `(num_filters, num_order + 1)`.
                                Lower delays coefficients are first, e.g. ``[a0, a1, a2, ...]``.
                                Must be same size as b_coeffs (pad with 0's as necessary).
        b_coeffs (Tensor): numerator coefficients of difference equation of dimension of either
                                1D with shape `(num_order + 1)` or 2D with shape `(num_filters, num_order + 1)`.
                                Lower delays coefficients are first, e.g. ``[b0, b1, b2, ...]``.
                                Must be same size as a_coeffs (pad with 0's as necessary).
        clamp (bool, optional): If ``True``, clamp the output signal to be in the range [-1, 1] (Default: ``True``)
        batching (bool, optional): Effective only when coefficients are 2D. If ``True``, then waveform should be at
                                    least 2D, and the size of second axis from last should equals to ``num_filters``.
                                    The output can be expressed as ``output[..., i, :] = lfilter(waveform[..., i, :],
                                    a_coeffs[i], b_coeffs[i], clamp=clamp, batching=False)``. (Default: ``True``)

    Returns:
        Tensor: Waveform with dimension of either `(..., num_filters, time)` if ``a_coeffs`` and ``b_coeffs``
        are 2D Tensors, or `(..., time)` otherwise.
    """
    if a_coeffs.size() != b_coeffs.size():
        raise ValueError(
            "Expected coeffs to be the same size."
            f"Found: a_coeffs size: {a_coeffs.size()}, b_coeffs size: {b_coeffs.size()}"
        )
    if a_coeffs.ndim > 2:
        raise ValueError(f"Expected coeffs to have greater than 1 dimension. Found: {a_coeffs.ndim}")

    if a_coeffs.ndim > 1:
        if batching:
            if waveform.ndim <= 0:
                raise ValueError("Expected waveform to have a positive number of dimensions." f"Found: {waveform.ndim}")
            if waveform.shape[-2] != a_coeffs.shape[0]:
                raise ValueError(
                    "Expected number of batches in waveform and coeffs to be the same."
                    f"Found: coeffs batches: {a_coeffs.shape[0]}, waveform batches: {waveform.shape[-2]}"
                )
        else:
            waveform = torch.stack([waveform] * a_coeffs.shape[0], -2)
    else:
        a_coeffs = a_coeffs.unsqueeze(0)
        b_coeffs = b_coeffs.unsqueeze(0)

    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, a_coeffs.shape[0], shape[-1])
    output = _lfilter(waveform, a_coeffs, b_coeffs)

    if clamp:
        output = torch.clamp(output, min=-1.0, max=1.0)

    # unpack batch
    output = output.reshape(shape[:-1] + output.shape[-1:])

    return output


def _get_sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    gcd: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: Optional[float] = None,
    device: torch.device = torch.device("cpu"),
    dtype: Optional[torch.dtype] = None,
):

    if not (int(orig_freq) == orig_freq and int(new_freq) == new_freq):
        raise Exception(
            "Frequencies must be of integer type to ensure quality resampling computation. "
            "To work around this, manually convert both frequencies to integer values "
            "that maintain their resampling rate ratio before passing them into the function. "
            "Example: To downsample a 44100 hz waveform by a factor of 8, use "
            "`orig_freq=8` and `new_freq=1` instead of `orig_freq=44100` and `new_freq=5512.5`. "
            "For more information, please refer to https://github.com/pytorch/audio/issues/1487."
        )

    if resampling_method in ["sinc_interpolation", "kaiser_window"]:
        method_map = {
            "sinc_interpolation": "sinc_interp_hann",
            "kaiser_window": "sinc_interp_kaiser",
        }
        warnings.warn(
            f'"{resampling_method}" resampling method name is being deprecated and replaced by '
            f'"{method_map[resampling_method]}" in the next release. '
            "The default behavior remains unchanged."
        )
    elif resampling_method not in ["sinc_interp_hann", "sinc_interp_kaiser"]:
        raise ValueError("Invalid resampling method: {}".format(resampling_method))

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    if lowpass_filter_width <= 0:
        raise ValueError("Low pass filter width should be positive.")
    base_freq = min(orig_freq, new_freq)
    # This will perform antialiasing filtering by removing the highest frequencies.
    # At first I thought I only needed this when downsampling, but when upsampling
    # you will get edge artifacts without this, as the edge is equivalent to zero padding,
    # which will add high freq artifacts.
    base_freq *= rolloff

    # The key idea of the algorithm is that x(t) can be exactly reconstructed from x[i] (tensor)
    # using the sinc interpolation formula:
    #   x(t) = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - t))
    # We can then sample the function x(t) with a different sample rate:
    #    y[j] = x(j / new_freq)
    # or,
    #    y[j] = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))

    # We see here that y[j] is the convolution of x[i] with a specific filter, for which
    # we take an FIR approximation, stopping when we see at least `lowpass_filter_width` zeros crossing.
    # But y[j+1] is going to have a different set of weights and so on, until y[j + new_freq].
    # Indeed:
    # y[j + new_freq] = sum_i x[i] sinc(pi * orig_freq * ((i / orig_freq - (j + new_freq) / new_freq))
    #                 = sum_i x[i] sinc(pi * orig_freq * ((i - orig_freq) / orig_freq - j / new_freq))
    #                 = sum_i x[i + orig_freq] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))
    # so y[j+new_freq] uses the same filter as y[j], but on a shifted version of x by `orig_freq`.
    # This will explain the F.conv1d after, with a stride of orig_freq.
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    # If orig_freq is still big after GCD reduction, most filters will be very unbalanced, i.e.,
    # they will have a lot of almost zero values to the left or to the right...
    # There is probably a way to evaluate those filters more efficiently, but this is kept for
    # future work.
    idx_dtype = dtype if dtype is not None else torch.float64

    idx = torch.arange(-width, width + orig_freq, dtype=idx_dtype, device=device)[None, None] / orig_freq

    t = torch.arange(0, -new_freq, -1, dtype=dtype, device=device)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)

    # we do not use built in torch windows here as we need to evaluate the window
    # at specific positions, not over a regular grid.
    if resampling_method == "sinc_interp_hann":
        window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    else:
        # sinc_interp_kaiser
        if beta is None:
            beta = 14.769656459379492
        beta_tensor = torch.tensor(float(beta))
        window = torch.i0(beta_tensor * torch.sqrt(1 - (t / lowpass_filter_width) ** 2)) / torch.i0(beta_tensor)

    t *= math.pi

    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * scale

    if dtype is None:
        kernels = kernels.to(dtype=torch.float32)

    return kernels, width


def _apply_sinc_resample_kernel(
    waveform: Tensor,
    orig_freq: int,
    new_freq: int,
    gcd: int,
    kernel: Tensor,
    width: int,
):
    if not waveform.is_floating_point():
        raise TypeError(f"Expected floating point type for waveform tensor, but received {waveform.dtype}.")

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    # pack batch
    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])

    num_wavs, length = waveform.shape
    waveform = torch.nn.functional.pad(waveform, (width, width + orig_freq))
    resampled = torch.nn.functional.conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    target_length = int(math.ceil(new_freq * length / orig_freq))
    resampled = resampled[..., :target_length]

    # unpack batch
    resampled = resampled.view(shape[:-1] + resampled.shape[-1:])
    return resampled
