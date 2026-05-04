import torch
from torch.utils.data import DataLoader
from dataset import WeatherDataset
from model import SimpleRestormer
from tqdm import tqdm
import time

# -------------------
# GPU CHECK
# -------------------
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------
# DATASET
# -------------------
train_dataset = WeatherDataset("dataset/train/input", "dataset/train/target")
val_dataset = WeatherDataset("dataset/val/input", "dataset/val/target")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# -------------------
# MODEL
# -------------------
model = SimpleRestormer().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.L1Loss()

# -------------------
# TRAINING LOOP
# -------------------
for epoch in range(1):
    start_time = time.time()

    model.train()
    train_loss = 0

    for i, (inp, tar) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        inp, tar = inp.to(device), tar.to(device)

        pred, _ = model(inp)
        loss = loss_fn(pred, tar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumulate correctly
        train_loss += loss.item() * inp.size(0)

        # print step loss
        if i % 50 == 0:
            print(f"Step {i}, Loss: {loss.item():.4f}")

    # average train loss
    train_loss = train_loss / len(train_loader.dataset)

    # -------------------
    # VALIDATION
    # -------------------
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for inp, tar in val_loader:
            inp, tar = inp.to(device), tar.to(device)

            pred, _ = model(inp)
            loss = loss_fn(pred, tar)

            val_loss += loss.item() * inp.size(0)

    # average val loss
    val_loss = val_loss / len(val_loader.dataset)

    end_time = time.time()

    print(f"\nEpoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Epoch Time: {end_time - start_time:.2f} seconds\n")
    torch.save(model.state_dict(), "model.pth")