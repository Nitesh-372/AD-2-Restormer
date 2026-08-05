import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import WeatherDataset
from model import AD2Restormer
from tqdm import tqdm

# -------------------
# LOSS FUNCTIONS
# -------------------
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (Smooth L1 Loss variant for Image Restoration)."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)

class EdgeLoss(nn.Module):
    """Sobel Edge-Preserving Supervision Loss."""
    def __init__(self, device="cpu"):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        self.sobel_x = sobel_x.repeat(3, 1, 1, 1)
        self.sobel_y = sobel_y.repeat(3, 1, 1, 1)

    def forward(self, x, y):
        grad_x_pred = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        grad_y_pred = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        grad_x_gt = F.conv2d(y, self.sobel_x, padding=1, groups=3)
        grad_y_gt = F.conv2d(y, self.sobel_y, padding=1, groups=3)

        loss_x = F.l1_loss(grad_x_pred, grad_x_gt)
        loss_y = F.l1_loss(grad_y_pred, grad_y_gt)
        return loss_x + loss_y

class FFTLoss(nn.Module):
    """Frequency Domain (FFT) Loss."""
    def forward(self, x, y):
        fft_x = torch.fft.rfft2(x, norm="backward")
        fft_y = torch.fft.rfft2(y, norm="backward")
        return F.l1_loss(torch.abs(fft_x), torch.abs(fft_y))

class SSIMLoss(nn.Module):
    """Differentiable Structural Similarity (SSIM) Loss."""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, img1, img2):
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        pad = self.window_size // 2

        mu1 = F.avg_pool2d(img1, self.window_size, stride=1, padding=pad)
        mu2 = F.avg_pool2d(img2, self.window_size, stride=1, padding=pad)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(img1 * img1, self.window_size, stride=1, padding=pad) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, self.window_size, stride=1, padding=pad) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, self.window_size, stride=1, padding=pad) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return 1.0 - ssim_map.mean()

def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio (PSNR)."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

# -------------------
# MAIN TRAINING PIPELINE
# -------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Dataset & Loaders
    train_dataset = WeatherDataset("dataset/train/input", "dataset/train/target", image_size=128, is_train=True)
    val_dataset = WeatherDataset("dataset/val/input", "dataset/val/target", image_size=256, is_train=False)

    # Multi-worker DataLoader with pin_memory to eliminate CPU bottlenecks
    num_workers = 2 if os.name == 'nt' else 4
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))

    # Initialize AD2Restormer Model
    model = AD2Restormer(dim=48, num_blocks=[2, 2, 2]).to(device)

    epochs = 30

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Loss Functions
    charbonnier_loss = CharbonnierLoss().to(device)
    edge_loss = EdgeLoss(device=device)
    fft_loss = FFTLoss().to(device)
    ssim_loss = SSIMLoss().to(device)

    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    best_psnr = 0.0

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        for i, (inp, tar) in enumerate(tqdm(train_loader, desc="Training")):
            inp, tar = inp.to(device), tar.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                pred, deg_logits, _ = model(inp)
                
                # Combined Loss
                l_charb = charbonnier_loss(pred, tar)
                l_edge = edge_loss(pred, tar)
                l_fft = fft_loss(pred, tar)
                l_ssim = ssim_loss(pred, tar)
                
                loss = l_charb + 0.05 * l_edge + 0.1 * l_fft + 0.2 * l_ssim

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inp.size(0)

            if (i + 1) % 100 == 0:
                print(f"Step [{i+1}/{len(train_loader)}] | Total Loss: {loss.item():.4f} (Charb: {l_charb.item():.4f}, Edge: {l_edge.item():.4f}, FFT: {l_fft.item():.4f}, SSIM: {l_ssim.item():.4f})")

        train_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        # Validation Phase
        model.eval()
        val_psnr = 0.0
        val_loss = 0.0

        with torch.no_grad():
            for inp, tar in tqdm(val_loader, desc="Evaluating"):
                inp, tar = inp.to(device), tar.to(device)
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    pred, _, _ = model(inp)
                    l_charb = charbonnier_loss(pred, tar)
                    val_loss += l_charb.item() * inp.size(0)

                # Measure PSNR
                val_psnr += calculate_psnr(pred, tar) * inp.size(0)

        val_loss /= len(val_loader.dataset)
        val_psnr /= len(val_loader.dataset)
        elapsed_time = time.time() - start_time

        print(f"Epoch {epoch+1} Completed in {elapsed_time:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f} dB")

        # Save model
        torch.save(model.state_dict(), "model.pth")
        print(" latest Model state saved to model.pth")
        # Save best model based on validation PSNR
        if val_psnr > best_psnr:
           best_psnr = val_psnr
           torch.save(model.state_dict(), "model_best.pth")
           print(f"Best model saved to model_best.pth with PSNR: {best_psnr:.2f} dB")

if __name__ == "__main__":
    train()