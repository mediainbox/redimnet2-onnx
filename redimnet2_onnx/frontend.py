import math

import numpy as np
import torch  # pylint: disable=import-error
from torch.nn import functional  # pylint: disable=import-error

SAMPLE_RATE = 16_000
MEL_BINS = 72
FFT_LENGTH = 512
FRAME_LENGTH = 400
FRAME_STEP = 160
LOW_FREQUENCY = 20
HIGH_FREQUENCY = 7_600
MIN_SECONDS = 1
MAX_SECONDS = 30
TIME_ALIGNMENT = 4
EPSILON = 1e-8


def aligned_frame_count(frame_count: int) -> int:
    if frame_count < TIME_ALIGNMENT:
        raise ValueError(f"mel input requires at least {TIME_ALIGNMENT} frames")
    return (frame_count // TIME_ALIGNMENT) * TIME_ALIGNMENT


def align_mel_time(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 4:
        raise ValueError("mel features must have shape [B, 1, 72, T]")
    return features[..., : aligned_frame_count(features.shape[-1])]


def _hz_to_mel(values: np.ndarray | float) -> np.ndarray:
    return 2_595 * np.log10(1 + np.asarray(values) / 700.0)


def _mel_filterbank() -> np.ndarray:
    points = np.linspace(_hz_to_mel(LOW_FREQUENCY), _hz_to_mel(HIGH_FREQUENCY), MEL_BINS + 2)
    lower = points[:-2].reshape(1, -1)
    center = points[1:-1].reshape(1, -1)
    upper = points[2:].reshape(1, -1)
    bins = _hz_to_mel(np.linspace(0, SAMPLE_RATE // 2, FFT_LENGTH // 2))[1:].reshape(-1, 1)
    filters = np.maximum(0.0, np.minimum((bins - lower) / (center - lower), (upper - bins) / (upper - center)))
    return np.vstack([np.zeros((1, MEL_BINS)), filters]).astype(np.float32)


def _dft_kernels() -> tuple[np.ndarray, np.ndarray]:
    samples = np.arange(FFT_LENGTH)
    frequencies = np.arange(FFT_LENGTH)
    phase = 2 * math.pi * samples[:, None] * frequencies[None, :] / FFT_LENGTH
    window = np.hamming(FRAME_LENGTH).astype(np.float32)
    real = np.cos(phase).astype(np.float32)[:FRAME_LENGTH, : FFT_LENGTH // 2]
    imaginary = np.sin(phase).astype(np.float32)[:FRAME_LENGTH, : FFT_LENGTH // 2]
    real *= window[:, None]
    imaginary *= window[:, None]
    return real.T[:, None, :], imaginary.T[:, None, :]


class ReDimNetFrontend(torch.nn.Module):
    """Exact FP32 waveform frontend used by upstream ReDimNet2."""

    def __init__(self) -> None:
        super().__init__()
        real, imaginary = _dft_kernels()
        self.register_buffer("real_kernel", torch.from_numpy(real))
        self.register_buffer("imaginary_kernel", torch.from_numpy(imaginary))
        self.register_buffer("preemphasis", torch.tensor([-0.97, 1.0], dtype=torch.float32).reshape(1, 1, 2))
        self.register_buffer("mel_filterbank", torch.from_numpy(_mel_filterbank().T[:, :, None]))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        if waveforms.ndim != 2:
            raise ValueError("waveforms must have shape [B, samples]")
        if waveforms.shape[0] != 1:
            raise ValueError("exported models currently require batch size 1")
        if waveforms.shape[-1] < MIN_SECONDS * SAMPLE_RATE:
            raise ValueError("models require at least 1 second of 16 kHz audio")
        if waveforms.shape[-1] > MAX_SECONDS * SAMPLE_RATE:
            raise ValueError("models accept at most 30 seconds of audio")

        with torch.inference_mode(), torch.autocast(waveforms.device.type, enabled=False):
            values = waveforms.float().unsqueeze(1)
            values = (values - values.mean(dim=2, keepdim=True)) / (
                values.std(dim=2, keepdim=True, unbiased=False) + EPSILON
            )
            values = functional.pad(values, (1, 0), mode="reflect")
            values = functional.conv1d(values, self.preemphasis)
            real = functional.conv1d(values, self.real_kernel, stride=FRAME_STEP, padding=FRAME_STEP // 2)
            imaginary = functional.conv1d(values, self.imaginary_kernel, stride=FRAME_STEP, padding=FRAME_STEP // 2)
            power = (real.square() + imaginary.square()).clamp(EPSILON, 1 / EPSILON)
            mel = functional.conv1d(power, self.mel_filterbank).clamp(EPSILON, 1 / EPSILON)
            mel = (mel + EPSILON).log()
            mel = mel - mel.mean(dim=-1, keepdim=True)
            return align_mel_time(mel.unsqueeze(1).float()).contiguous()
