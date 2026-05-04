import torch
import cv2
import numpy as np
from model import SimpleRestormer

# -------------------
# LOAD MODEL
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleRestormer().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# -------------------
# LOAD IMAGE
# -------------------
img_path = "dataset\\full\\input\\city_read_12537.jpg"   # put your image here

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# resize (same as training)
img = cv2.resize(img, (256, 256))

# normalize
img = img / 255.0
img = np.transpose(img, (2, 0, 1))
img = torch.tensor(img).float().unsqueeze(0).to(device)

# -------------------
# INFERENCE
# -------------------
with torch.no_grad():
    output, _ = model(img)

# -------------------
# SAVE OUTPUT
# -------------------
output = output.squeeze(0).cpu().numpy()
output = np.transpose(output, (1, 2, 0))

# clip values
output = np.clip(output, 0, 1)

output = (output * 255).astype(np.uint8)

cv2.imwrite("output.png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

print("Output saved as output.png")