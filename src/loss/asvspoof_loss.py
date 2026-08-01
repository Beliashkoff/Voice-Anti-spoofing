from torch import nn


class ASVspoofLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels, **batch):
        labels = labels.float().view(-1, 1)
        return {"loss": self.loss(logits, labels)}
