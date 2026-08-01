import torch
import torchaudio
from torch import nn


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        win_length=512,
        hop_length=128,
        n_mels=128,
        eps=1e-6,
    ):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.eps = eps

    def forward(self, x):
        x = self.mel(x)
        return torch.log(x.clamp_min(self.eps))
