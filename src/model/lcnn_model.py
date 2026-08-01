from torch import nn


class MFM(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        a, b = x.chunk(2, dim=self.dim)
        return a.max(b)


class ConvMFM(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size, stride, padding)
        self.mfm = MFM(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.mfm(x)
        return x


class LinearMFM(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.fc = nn.Linear(in_feat, out_feat * 2)
        self.mfm = MFM(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.mfm(x)
        return x


class LCNNModel(nn.Module):
    def __init__(
        self,
        n_class=2,
        dropout=0.5,
        feature_dropout=0.1,
        pooled_size=8,
        embedding_dim=64,
    ):
        super().__init__()

        self.conv1 = ConvMFM(1, 32, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2a = ConvMFM(32, 32, kernel_size=1)
        self.bn2a = nn.BatchNorm2d(32)
        self.conv2 = ConvMFM(32, 48, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(48)

        self.conv3a = ConvMFM(48, 48, kernel_size=1)
        self.bn3a = nn.BatchNorm2d(48)
        self.conv3 = ConvMFM(48, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4a = ConvMFM(64, 64, kernel_size=1)
        self.bn4a = nn.BatchNorm2d(64)
        self.conv4 = ConvMFM(64, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv5a = ConvMFM(32, 32, kernel_size=1)
        self.bn5a = nn.BatchNorm2d(32)
        self.conv5 = ConvMFM(32, 32, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.feature_dropout = nn.Dropout2d(feature_dropout)

        self.avgpool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))

        self.fc1 = LinearMFM(32 * pooled_size * pooled_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn6 = nn.BatchNorm1d(embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, n_class)

    def forward(self, data_object, **batch):
        x = data_object
        batch_size = x.shape[0]
        segment_count = 1
        if x.ndim == 5:
            segment_count = x.shape[1]
            x = x.flatten(0, 1)
        elif x.ndim != 4:
            raise ValueError(f"неверная форма входа {x.shape}")

        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.bn2a(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)

        x = self.conv3a(x)
        x = self.bn3a(x)
        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.bn4a(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = self.conv5a(x)
        x = self.bn5a(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = self.feature_dropout(x)

        x = self.avgpool(x)
        x = x.flatten(1)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.bn6(x)
        x = self.fc2(x)

        segment_logits = x.view(batch_size, segment_count, -1)
        logits = segment_logits.mean(dim=1)
        return {"logits": logits, "segment_logits": segment_logits}

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
