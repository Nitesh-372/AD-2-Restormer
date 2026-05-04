import torch
import torch.nn as nn


class AD2Module(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        feat = self.conv(x)                      # [B, 64, 1, 1]
        feat_flat = feat.view(feat.size(0), -1)  # [B, 64]
        out = self.fc(feat_flat)                 # degradation prediction
        return out, feat_flat


class SimpleRestormer(nn.Module):
    def __init__(self):
        super().__init__()

        self.ad2 = AD2Module()

        # ✅ Conv projection instead of Linear
        self.feature_proj = nn.Conv2d(64, 128, kernel_size=1)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x):
        degradation, feat = self.ad2(x)

        enc = self.encoder(x)  # [B, 128, H, W]

        # ✅ reshape + project
        feat = feat.view(feat.size(0), 64, 1, 1)   # [B, 64, 1, 1]
        feat = self.feature_proj(feat)             # [B, 128, 1, 1]

        # ✅ better conditioning (attention-like)
        enc = enc * (1 + feat)

        out = self.decoder(enc)

        return out, degradation