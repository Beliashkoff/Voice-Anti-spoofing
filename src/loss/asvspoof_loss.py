from torch import nn


class ASVspoofLoss(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, labels, **batch):
        return {"loss": self.loss(logits, labels)}
