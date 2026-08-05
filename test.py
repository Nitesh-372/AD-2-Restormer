import os
import time
import math
import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from model import AD2Restormer

torch.backends.cudnn.benchmark = True


def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * math.log10(255.0 / math.sqrt(mse))


def make_positions(length, tile_size, stride):
    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def make_blend_weight(tile_size, overlap, device):
    weight_1d = torch.ones(tile_size, dtype=torch.float32, device=device)
    ramp = min(overlap, tile_size // 2)
    if ramp > 0:
        values = torch.linspace(0.1, 1.0, ramp, dtype=torch.float32, device=device)
        weight_1d[:ramp] = values
        weight_1d[-ramp:] = values.flip(0)
    return weight_1d.view(1, 1, tile_size, 1) * weight_1d.view(1, 1, 1, tile_size)


def to_tensor(rgb_img, device):
    tensor = torch.from_numpy(rgb_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor.to(device)
    if device == "cuda":
        tensor = tensor.to(memory_format=torch.channels_last)
    return tensor


def run_model(model, img_tensor, device):
    with torch.amp.autocast(device, enabled=(device == "cuda")):
        output_tensor, _, _ = model(img_tensor)
    return output_tensor.float()


def infer_resized(model, img_rgb, args, device):
    h_orig, w_orig = img_rgb.shape[:2]
    model_input_rgb = cv2.resize(img_rgb, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
    img_tensor = to_tensor(model_input_rgb, device)
    output_tensor = run_model(model, img_tensor, device)
    output_np = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    return cv2.resize(output_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)


def infer_full_res(model, img_rgb, device):
    h_orig, w_orig = img_rgb.shape[:2]
    factor = 8
    h_pad = (factor - h_orig % factor) % factor
    w_pad = (factor - w_orig % factor) % factor

    img_tensor = to_tensor(img_rgb, device)
    if h_pad > 0 or w_pad > 0:
        img_tensor = F.pad(img_tensor, (0, w_pad, 0, h_pad), mode="reflect")

    output_tensor = run_model(model, img_tensor, device)
    output_np = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    return output_np[:h_orig, :w_orig, :]


def infer_tiled(model, img_rgb, args, device):
    h_orig, w_orig = img_rgb.shape[:2]
    tile_size = args.tile_size
    overlap = args.overlap
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("--overlap must be smaller than --tile-size")

    pad_h = max(tile_size - h_orig, 0)
    pad_w = max(tile_size - w_orig, 0)
    if pad_h > 0 or pad_w > 0:
        img_rgb = np.pad(img_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    h_pad, w_pad = img_rgb.shape[:2]
    y_positions = make_positions(h_pad, tile_size, stride)
    x_positions = make_positions(w_pad, tile_size, stride)

    output_accum = torch.zeros((1, 3, h_pad, w_pad), dtype=torch.float32, device=device)
    weight_accum = torch.zeros((1, 1, h_pad, w_pad), dtype=torch.float32, device=device)
    blend_weight = make_blend_weight(tile_size, overlap, device)

    for y in y_positions:
        for x in x_positions:
            tile_rgb = img_rgb[y:y + tile_size, x:x + tile_size, :]
            tile_tensor = to_tensor(tile_rgb, device)
            tile_output = run_model(model, tile_tensor, device)
            output_accum[:, :, y:y + tile_size, x:x + tile_size] += tile_output * blend_weight
            weight_accum[:, :, y:y + tile_size, x:x + tile_size] += blend_weight

    output_tensor = output_accum / weight_accum.clamp_min(1e-6)
    output_np = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    return output_np[:h_orig, :w_orig, :], len(y_positions) * len(x_positions)


def run_inference(model, img_rgb, args, device):
    if args.tiled:
        return infer_tiled(model, img_rgb, args, device)
    if args.full_res:
        return infer_full_res(model, img_rgb, device), 1
    return infer_resized(model, img_rgb, args, device), 1


def main():
    parser = argparse.ArgumentParser(description="Test and Evaluate AD2 Restormer Model")
    parser.add_argument("--input", type=str, default="dataset/full/input/38_rain.png", help="Path to input degraded image")
    parser.add_argument("--target", type=str, default="dataset/full/target/38_rain.png", help="Path to ground truth clean target image (optional)")
    parser.add_argument("--output", type=str, default="output.png", help="Path where restored output image will be saved")
    parser.add_argument("--weights", type=str, default="model.pth", help="Path to model weights file")
    parser.add_argument("--size", type=int, default=256, help="Model input size used for resized inference")
    parser.add_argument("--runs", type=int, default=30, help="Number of timed inference runs")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup inference runs")
    parser.add_argument("--full-res", action="store_true", help="Run inference on the original image size instead of --size")
    parser.add_argument("--tiled", action="store_true", help="Run full-image inference using overlapping tiles")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size for --tiled inference")
    parser.add_argument("--overlap", type=int, default=32, help="Tile overlap in pixels for --tiled inference")
    args = parser.parse_args()

    if args.tiled and args.full_res:
        raise ValueError("Use either --tiled or --full-res, not both")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AD2Restormer(dim=48, num_blocks=[2, 2, 2]).to(device)

    if os.path.exists(args.weights):
        try:
            state_dict = torch.load(args.weights, map_location=device, weights_only=True)
            if "shallow_conv.weight" in state_dict:
                model.load_state_dict(state_dict)
                print(f"Successfully loaded model weights from: {args.weights}")
            else:
                print(f"Weights in '{args.weights}' are from an older architecture. Running with fresh initialized weights...")
        except Exception as e:
            print(f"Loading weights error: {e}")
    else:
        print(f"Weights file '{args.weights}' not found. Running inference with initialized architecture...")

    model.eval()
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if device == "cuda" and os.name != "nt":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile() applied for faster inference.")
        except Exception:
            print("torch.compile() not available, skipping.")
    elif os.name == "nt":
        print("torch.compile() skipped (not supported on Windows). FP16 autocast active.")

    input_path = args.input
    if not os.path.exists(input_path):
        fallback_dir = "dataset/full/input"
        if os.path.exists(fallback_dir) and len(os.listdir(fallback_dir)) > 0:
            first_file = os.listdir(fallback_dir)[0]
            input_path = os.path.join(fallback_dir, first_file)
            print(f"Warning: Specified input not found. Falling back to: {input_path}")
        else:
            raise FileNotFoundError(f"Input image not found: {args.input}")

    print(f"Processing image: {input_path}")

    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]

    with torch.inference_mode():
        for _ in range(max(0, args.warmup)):
            _, tile_count = run_inference(model, img_rgb, args, device)
    if device == "cuda":
        torch.cuda.synchronize()

    num_runs = max(1, args.runs)
    start_time = time.perf_counter()
    with torch.inference_mode():
        for _ in range(num_runs):
            output_np, tile_count = run_inference(model, img_rgb, args, device)
    if device == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    avg_latency_ms = (total_time / num_runs) * 1000
    fps = num_runs / total_time

    output_restored_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, output_restored_bgr)
    print(f"Restored image saved to: {args.output}")

    if args.tiled:
        inference_size = f"tiled {args.tile_size}x{args.tile_size}, overlap {args.overlap}px"
    elif args.full_res:
        inference_size = "full resolution"
    else:
        inference_size = f"{args.size}x{args.size}"

    print("\n================ BENCHMARK & TEST RESULTS ================")
    print(f"Input Image     : {input_path}")
    print(f"Original Size   : {w_orig}x{h_orig} px")
    print(f"Inference Size  : {inference_size}")
    print(f"Tiles Per Image : {tile_count}")
    print(f"Timed Runs      : {num_runs}")
    print(f"Output Image    : {args.output}")
    print(f"Average Latency : {avg_latency_ms:.2f} ms")
    print(f"Throughput      : {fps:.2f} FPS")

    target_path = args.target
    if os.path.exists(target_path):
        target_bgr = cv2.imread(target_path)
        if target_bgr is not None:
            if target_bgr.shape[:2] != output_restored_bgr.shape[:2]:
                target_bgr = cv2.resize(target_bgr, (output_restored_bgr.shape[1], output_restored_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
            psnr_val = calculate_psnr(output_restored_bgr, target_bgr)
            print(f"Quality (PSNR)  : {psnr_val:.2f} dB")
    print("==========================================================\n")


if __name__ == "__main__":
    main()
