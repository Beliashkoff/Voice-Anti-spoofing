import torch
import torchaudio
from torch import nn


class STFT(nn.Module):
    def __init__(
        self,
        n_fft=512,
        hop_length=128,
        win_length=512,
        power=2.0,
        eps=1e-6,
        normalize=True,
    ):
        super().__init__()
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=power,
        )
        self.eps = eps
        self.normalize = normalize

    def forward(self, x):
        x = self.spec(x)
        x = torch.log(x.clamp_min(self.eps))
        if self.normalize:
            mean = x.mean(dim=(-2, -1), keepdim=True)
            std = x.std(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
            x = (x - mean) / std
        return x
