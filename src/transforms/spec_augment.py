import torch
from torch import nn


class SpecAugment(nn.Module):
    def __init__(self, freq_width=16, time_width=32, p=0.7):
        super().__init__()
        self.freq_width = freq_width
        self.time_width = time_width
        self.p = p

    def forward(self, x):
        if not self.training or torch.rand(1, device=x.device).item() > self.p:
            return x

        result = x.clone()
        freq_size = result.shape[-2]
        time_size = result.shape[-1]

        freq_width = min(self.freq_width, freq_size)
        time_width = min(self.time_width, time_size)

        if freq_width > 0:
            width = torch.randint(freq_width + 1, (1,), device=x.device).item()
            if width > 0:
                start = torch.randint(freq_size - width + 1, (1,), device=x.device).item()
                result[..., start : start + width, :] = 0

        if time_width > 0:
            width = torch.randint(time_width + 1, (1,), device=x.device).item()
            if width > 0:
                start = torch.randint(time_size - width + 1, (1,), device=x.device).item()
                result[..., start : start + width] = 0

        return result
