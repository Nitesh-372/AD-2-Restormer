import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    """Channel-wise Layer Normalization for 2D Spatial Feature Maps."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias

class DegradationEncoder(nn.Module):
    """Dynamic Degradation Token Generator (AD2 Branch)"""
    def __init__(self, in_channels=3, embed_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, embed_dim, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        feat = self.conv(x).view(x.size(0), -1)  # [B, embed_dim]
        logits = self.classifier(feat)           # [B, 2] classification
        return feat, logits

class FiLMConditioning(nn.Module):
    """Feature-wise Linear Modulation with Zero-Initialized Residual Scaling."""
    def __init__(self, token_dim=64, num_features=48):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, num_features * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.gamma_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, token):
        # x: [B, C, H, W], token: [B, token_dim]
        params = self.mlp(token).unsqueeze(-1).unsqueeze(-1)  # [B, 2C, 1, 1]
        gamma, beta = torch.chunk(params, 2, dim=1)
        # Residual gating for training stability
        return x + self.gamma_bias * (x * gamma + beta)

class MDTA(nn.Module):
    """Multi-DConv Head Transposed Self-Attention (Channel-wise Self Attention)."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.num_heads, c // self.num_heads, h * w)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.view(b, c, h, w)
        out = self.project_out(out)
        return out

class GDFN(nn.Module):
    """Gated-DConv Feed-Forward Network."""
    def __init__(self, dim, ffn_expansion_factor=2.66):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class RestormerBlock(nn.Module):
    """Core Restormer Transformer Block: LayerNorm -> MDTA -> LayerNorm -> GDFN."""
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.66):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class DualAttentionBottleneck(nn.Module):
    """Channel + Spatial Dual Attention Bottleneck Block."""
    def __init__(self, dim):
        super().__init__()
        # Channel Attention
        self.ca_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_mlp = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.sa_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca_weight = self.ca_mlp(self.ca_pool(x))
        x_ca = x * ca_weight

        # Spatial attention
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        sa_weight = self.sa_conv(torch.cat([avg_out, max_out], dim=1))
        out = x_ca * sa_weight
        return out

class AD2Restormer(nn.Module):
    """
    High-Performance Real-Time Vision Restormer Architecture.
    Includes Dynamic Degradation Token T_d, FiLM Conditioning, 3-Stage U-Net Restormer,
    Dual-Attention Bottleneck, and Global Residual Learning (I + R).
    """
    def __init__(self, in_channels=3, out_channels=3, dim=48, num_blocks=[2, 2, 2], num_heads=[1, 2, 4]):
        super().__init__()

        # Dynamic Degradation Token Predictor
        self.deg_encoder = DegradationEncoder(in_channels=in_channels, embed_dim=64)

        # Shallow Feature Extractor
        self.shallow_conv = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)

        # Stage 1 Encoder
        self.film1 = FiLMConditioning(token_dim=64, num_features=dim)
        self.enc1 = nn.Sequential(*[RestormerBlock(dim, num_heads[0]) for _ in range(num_blocks[0])])
        self.down1 = nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1)

        # Stage 2 Encoder
        self.film2 = FiLMConditioning(token_dim=64, num_features=dim * 2)
        self.enc2 = nn.Sequential(*[RestormerBlock(dim * 2, num_heads[1]) for _ in range(num_blocks[1])])
        self.down2 = nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1)

        # Stage 3 Encoder
        self.film3 = FiLMConditioning(token_dim=64, num_features=dim * 4)
        self.enc3 = nn.Sequential(*[RestormerBlock(dim * 4, num_heads[2]) for _ in range(num_blocks[2])])

        # Dual-Attention Bottleneck
        self.bottleneck = DualAttentionBottleneck(dim * 4)

        # Stage 3 Decoder
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=2, stride=2),
            LayerNorm2d(dim * 2)
        )
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1)
        self.dec2 = nn.Sequential(*[RestormerBlock(dim * 2, num_heads[1]) for _ in range(num_blocks[1])])

        # Stage 2 Decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2),
            LayerNorm2d(dim)
        )
        self.reduce1 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.dec1 = nn.Sequential(*[RestormerBlock(dim, num_heads[0]) for _ in range(num_blocks[0])])

        # Residual Prediction Layer
        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)
        )

        # Zero-initialize residual prediction output to start from identity map
        nn.init.zeros_(self.residual_conv[-1].weight)
        nn.init.zeros_(self.residual_conv[-1].bias)

    def forward(self, x):
        # 1. Dynamic Degradation Token T_d & Classification Logits
        token_d, deg_logits = self.deg_encoder(x)

        # 2. Shallow Feature Extraction
        f0 = self.shallow_conv(x)

        # 3. Encoder Stage 1
        f1 = self.film1(f0, token_d)
        f1 = self.enc1(f1)

        # 4. Downsample to Stage 2
        f2_in = self.down1(f1)
        f2 = self.film2(f2_in, token_d)
        f2 = self.enc2(f2)

        # 5. Downsample to Stage 3
        f3_in = self.down2(f2)
        f3 = self.film3(f3_in, token_d)
        f3 = self.enc3(f3)

        # 6. Bottleneck Block
        b_feat = self.bottleneck(f3)

        # 7. Decoder Stage 3 (Upsample + Skip Connection)
        d2 = self.up2(b_feat)
        d2 = self.reduce2(torch.cat([d2, f2], dim=1))
        d2 = self.dec2(d2)

        # 8. Decoder Stage 2 (Upsample + Skip Connection)
        d1 = self.up1(d2)
        d1 = self.reduce1(torch.cat([d1, f1], dim=1))
        d1 = self.dec1(d1)

        # 9. Global Residual Prediction
        predicted_residual = self.residual_conv(d1)

        # 10. Global Residual Addition
        clean_out = x + predicted_residual

        return clean_out, deg_logits, token_d