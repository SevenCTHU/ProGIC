"""Evaluate ProGIC at its five progressive rate points."""
import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from ProGIC import ProGIC

ROOT = Path(__file__).resolve().parent


def load_image(path, device, resize=None):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if resize:
            image = image.resize((resize[1], resize[0]), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def pad_image(x):
    height, width = x.shape[-2:]
    return F.pad(x, (0, -width % 16, 0, -height % 16), value=0)


def pack_indices(indices, bits_per_index=10):
    """Fixed-length RVQ payload, MSB first. Shape metadata is kept separately."""
    values = torch.stack(indices).detach().cpu().numpy().astype(np.uint16).ravel()
    shifts = np.arange(bits_per_index - 1, -1, -1)
    bits = ((values[:, None] >> shifts) & 1).astype(np.uint8).ravel()
    return np.packbits(bits).tobytes(), (len(indices), *indices[0].shape)


def unpack_indices(payload, shape, device, bits_per_index=10):
    count = math.prod(shape)
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))[:count * bits_per_index]
    values = bits.reshape(count, bits_per_index) @ (1 << np.arange(bits_per_index - 1, -1, -1))
    tensors = torch.from_numpy(values.astype(np.int64).reshape(shape)).to(device)
    return list(tensors.unbind(0))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def reconstruct(net, frame, index):
    height, width = frame.shape[-2:]
    padded = pad_image(frame)
    synchronize(frame.device)
    start = time.perf_counter()
    indices = net.compress(padded, index)
    payload, shape = pack_indices(indices, net.bits_per_index)
    synchronize(frame.device)
    encode_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    decoded_indices = unpack_indices(payload, shape, frame.device, net.bits_per_index)
    reconstruction = net.decompress(decoded_indices, index)[..., :height, :width]
    synchronize(frame.device)
    decode_ms = (time.perf_counter() - start) * 1000
    bpp = len(payload) * 8 / (frame.shape[0] * height * width)
    return reconstruction, bpp, encode_ms, decode_ms


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("base", "small"), default="base")
    parser.add_argument("--kodak-dir", type=Path, default=ROOT / "kodak")
    parser.add_argument("--ckpt-path", type=Path, help="Defaults to the selected variant's checkpoint")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--index", type=int, nargs="+", choices=range(5), default=list(range(5)))
    parser.add_argument("--save-dir", type=Path, help="Save reconstructed PNGs in per-rate folders")
    parser.add_argument("--csv", type=Path, help="Write per-image results, replacing an existing CSV")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), help="Optional resize; default keeps native resolution")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N images")
    parser.add_argument("--skip-perceptual", action="store_true", help="Run PSNR, BPP and timing without LPIPS/DISTS weights")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU thread count")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.resize and min(args.resize) < 1:
        parser.error("--resize dimensions must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    checkpoint = args.ckpt_path or ROOT / "ckpt" / ("ckpt_s.pth.tar" if args.variant == "small" else "ckpt.pth.tar")
    net = ProGIC(args.variant).to(device).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net.load_state_dict(state.get("state_dict", state), strict=True)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    files = sorted(p for p in args.kodak_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise ValueError(f"No images found in {args.kodak_dir}")

    lpips_fn = dists_fn = None
    if not args.skip_perceptual:
        import lpips
        from DISTS_pytorch import DISTS
        lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
        dists_fn = DISTS().to(device).eval()

    print(f"ProGIC {args.variant} | {device} | {len(files)} images | {sum(p.numel() for p in net.parameters()) / 1e6:.3f} M parameters")
    print(f"Checkpoint: {checkpoint}")
    print("BPP counts packed index bytes, excluding shape metadata. Timing includes pack/unpack and device transfers.")
    rows = []
    for index in args.index:
        # Warm up separately for each rate, outside measured inference.
        warmup = pad_image(load_image(files[0], device, args.resize))
        net.decompress(net.compress(warmup, index), index)
        synchronize(device)
        rate_rows = []
        for path in files:
            frame = load_image(path, device, args.resize)
            reconstructed, bpp, encode_ms, decode_ms = reconstruct(net, frame, index)
            reconstructed = reconstructed.clamp(-1, 1)
            mse = F.mse_loss((reconstructed + 1) / 2, (frame + 1) / 2).item()
            psnr = -10 * math.log10(mse) if mse else float("inf")
            row = dict(image=path.name, index=index, stages=index + 1, height=frame.shape[-2], width=frame.shape[-1], encode_ms=encode_ms, decode_ms=decode_ms, psnr=psnr, bpp=bpp)
            if lpips_fn is not None:
                row["lpips"] = lpips_fn(reconstructed, frame).mean().item()
                row["dists"] = dists_fn((reconstructed + 1) / 2, (frame + 1) / 2).mean().item()
            if args.save_dir:
                destination = args.save_dir / f"rate_{index}"
                destination.mkdir(parents=True, exist_ok=True)
                pixels = ((reconstructed[0] + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
                Image.fromarray(pixels).save(destination / f"{path.stem}.png")
            rate_rows.append(row)
            print(f"[{index}] {path.name} | " + " | ".join(f"{key}={row[key]:.5f}" for key in ("encode_ms", "decode_ms", "psnr", "bpp", "lpips", "dists") if key in row))
        keys = ("encode_ms", "decode_ms", "psnr", "bpp", "lpips", "dists")
        print(f"AVG index={index} | " + " | ".join(f"{key}={np.mean([r[key] for r in rate_rows]):.5f}" for key in keys if key in rate_rows[0]))
        rows.extend(rate_rows)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
