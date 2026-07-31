from torch import nn


class ASVspoofLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels, **batch):
        return {"loss": self.loss(logits, labels)}
