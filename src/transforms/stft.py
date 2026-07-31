import torch
import torchaudio
from torch import nn


class STFT(nn.Module):
    def __init__(self, n_fft=1724, hop_length=130, win_length=1724):
        super().__init__()
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=2.0,
        )

    def forward(self, x):
        x = self.spec(x)
        x = torch.log(x + 1e-6)
        return x
