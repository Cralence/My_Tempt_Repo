import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import math

import torch
from torch import Tensor

import mmldm.audio.torchaudio_.functional as F


class Spectrogram(torch.nn.Module):
    r"""Create a spectrogram from a audio signal.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float or None, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc.
            If None, then the complex spectrum is returned instead. (Default: ``2``)
        normalized (bool or str, optional): Whether to normalize by magnitude after stft. If input is str, choices are
            ``"window"`` and ``"frame_length"``, if specific normalization type is desirable. ``True`` maps to
            ``"window"``. (Default: ``False``)
        wkwargs (dict or None, optional): Arguments for window function. (Default: ``None``)
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. (Default: ``"reflect"``)
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy (Default: ``True``)
        return_complex (bool, optional):
            Deprecated and not used.

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = torchaudio.transforms.Spectrogram(n_fft=800)
        >>> spectrogram = transform(waveform)

    """
    __constants__ = ["n_fft", "win_length", "hop_length", "pad", "power", "normalized"]

    def __init__(
        self,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        pad: int = 0,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        power: Optional[float] = 2.0,
        normalized: Union[bool, str] = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        return_complex: Optional[bool] = None,
    ) -> None:
        super(Spectrogram, self).__init__()
        torch._C._log_api_usage_once("torchaudio.transforms.Spectrogram")
        self.n_fft = n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequencies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer("window", window)
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        if return_complex is not None:
            warnings.warn(
                "`return_complex` argument is now deprecated and is not effective."
                "`torchaudio.transforms.Spectrogram(power=None)` always returns a tensor with "
                "complex dtype. Please remove the argument in the function call."
            )

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Dimension (..., freq, time), where freq is
            ``n_fft // 2 + 1`` where ``n_fft`` is the number of
            Fourier bins, and time is the number of window hops (n_frame).
        """
        return F.spectrogram(
            waveform,
            self.pad,
            self.window,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.power,
            self.normalized,
            self.center,
            self.pad_mode,
            self.onesided,
        )


class MelSpectrogram(torch.nn.Module):
    r"""Create MelSpectrogram for a raw audio signal.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    This is a composition of :py:func:`torchaudio.transforms.Spectrogram` and
    and :py:func:`torchaudio.transforms.MelScale`.

    Sources
        * https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
        * https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
        * http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``None``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
        normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)
        wkwargs (Dict[..., ...] or None, optional): Arguments for window function. (Default: ``None``)
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. (Default: ``"reflect"``)
        onesided: Deprecated and unused.
        norm (str or None, optional): If "slaney", divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.MelSpectrogram(sample_rate)
        >>> mel_specgram = transform(waveform)  # (channel, n_mels, time)

    See also:
        :py:func:`torchaudio.functional.melscale_fbanks` - The function used to
        generate the filter banks.
    """
    __constants__ = ["sample_rate", "n_fft", "win_length", "hop_length", "pad", "n_mels", "f_min"]

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 128,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: Optional[bool] = None,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ) -> None:
        super(MelSpectrogram, self).__init__()
        torch._C._log_api_usage_once("torchaudio.transforms.MelSpectrogram")

        if onesided is not None:
            warnings.warn(
                "Argument 'onesided' has been deprecated and has no influence on the behavior of this module."
            )

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.n_mels = n_mels  # number of mel frequency bins
        self.f_max = f_max
        self.f_min = f_min
        self.spectrogram = Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=self.pad,
            window_fn=window_fn,
            power=self.power,
            normalized=self.normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=True,
        )
        self.mel_scale = MelScale(
            self.n_mels, self.sample_rate, self.f_min, self.f_max, self.n_fft // 2 + 1, norm, mel_scale
        )

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """
        specgram = self.spectrogram(waveform)
        mel_specgram = self.mel_scale(specgram)
        return mel_specgram


class MelScale(torch.nn.Module):
    r"""Turn a normal STFT into a mel frequency STFT with triangular filter banks.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        n_stft (int, optional): Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`. (Default: ``201``)
        norm (str or None, optional): If ``"slaney"``, divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> spectrogram_transform = transforms.Spectrogram(n_fft=1024)
        >>> spectrogram = spectrogram_transform(waveform)
        >>> melscale_transform = transforms.MelScale(sample_rate=sample_rate, n_stft=1024 // 2 + 1)
        >>> melscale_spectrogram = melscale_transform(spectrogram)

    See also:
        :py:func:`torchaudio.functional.melscale_fbanks` - The function used to
        generate the filter banks.
    """
    __constants__ = ["n_mels", "sample_rate", "f_min", "f_max"]

    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_stft: int = 201,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ) -> None:
        super(MelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.norm = norm
        self.mel_scale = mel_scale

        if f_min > self.f_max:
            raise ValueError("Require f_min: {} <= f_max: {}".format(f_min, self.f_max))

        fb = F.melscale_fbanks(n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, self.norm, self.mel_scale)
        self.register_buffer("fb", fb)

    def forward(self, specgram: Tensor) -> Tensor:
        r"""
        Args:
            specgram (Tensor): A spectrogram STFT of dimension (..., freq, time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """

        # (..., time, freq) dot (freq, n_mels) -> (..., n_mels, time)
        mel_specgram = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)

        return mel_specgram


class Loudness(torch.nn.Module):
    r"""Measure audio loudness according to the ITU-R BS.1770-4 recommendation.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    Args:
        sample_rate (int): Sample rate of audio signal.

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.Loudness(sample_rate)
        >>> loudness = transform(waveform)

    Reference:
        - https://www.itu.int/rec/R-REC-BS.1770-4-201510-I/en
    """
    __constants__ = ["sample_rate"]

    def __init__(self, sample_rate: int):
        super(Loudness, self).__init__()
        self.sample_rate = sample_rate

    def forward(self, wavefrom: Tensor):
        r"""
        Args:
            waveform(torch.Tensor): audio waveform of dimension `(..., channels, time)`

        Returns:
            Tensor: loudness estimates (LKFS)
        """
        return F.loudness(wavefrom, self.sample_rate)


class Resample(torch.nn.Module):
    r"""Resample a signal from one frequency to another. A resampling method can be given.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Note:
        If resampling on waveforms of higher precision than float32, there may be a small loss of precision
        because the kernel is cached once as float32. If high precision resampling is important for your application,
        the functional form will retain higher precision, but run slower because it does not cache the kernel.
        Alternatively, you could rewrite a transform that caches a higher precision kernel.

    Args:
        orig_freq (int, optional): The original frequency of the signal. (Default: ``16000``)
        new_freq (int, optional): The desired frequency. (Default: ``16000``)
        resampling_method (str, optional): The resampling method to use.
            Options: [``sinc_interp_hann``, ``sinc_interp_kaiser``] (Default: ``"sinc_interp_hann"``)
        lowpass_filter_width (int, optional): Controls the sharpness of the filter, more == sharper
            but less efficient. (Default: ``6``)
        rolloff (float, optional): The roll-off frequency of the filter, as a fraction of the Nyquist.
            Lower values reduce anti-aliasing, but also reduce some of the highest frequencies. (Default: ``0.99``)
        beta (float or None, optional): The shape parameter used for kaiser window.
        dtype (torch.device, optional):
            Determnines the precision that resampling kernel is pre-computed and cached. If not provided,
            kernel is computed with ``torch.float64`` then cached as ``torch.float32``.
            If you need higher precision, provide ``torch.float64``, and the pre-computed kernel is computed and
            cached as ``torch.float64``. If you use resample with lower precision, then instead of providing this
            providing this argument, please use ``Resample.to(dtype)``, so that the kernel generation is still
            carried out on ``torch.float64``.

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.Resample(sample_rate, sample_rate/10)
        >>> waveform = transform(waveform)
    """

    def __init__(
        self,
        orig_freq: int = 16000,
        new_freq: int = 16000,
        resampling_method: str = "sinc_interp_hann",
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        beta: Optional[float] = None,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.gcd = math.gcd(int(self.orig_freq), int(self.new_freq))
        self.resampling_method = resampling_method
        self.lowpass_filter_width = lowpass_filter_width
        self.rolloff = rolloff
        self.beta = beta

        if self.orig_freq != self.new_freq:
            kernel, self.width = F._get_sinc_resample_kernel(
                self.orig_freq,
                self.new_freq,
                self.gcd,
                self.lowpass_filter_width,
                self.rolloff,
                self.resampling_method,
                beta,
                dtype=dtype,
            )
            self.register_buffer("kernel", kernel)

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Output signal of dimension (..., time).
        """
        if self.orig_freq == self.new_freq:
            return waveform
        return F._apply_sinc_resample_kernel(waveform, self.orig_freq, self.new_freq, self.gcd, self.kernel, self.width)
