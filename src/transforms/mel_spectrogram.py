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
    ):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
    def forward(self, x):
        return self.mel(x)
