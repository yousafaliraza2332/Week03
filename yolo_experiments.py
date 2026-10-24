
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

MODELS = ["yolo11n.pt", "yolo11x.pt"]
CONFS = [0.2, 0.4, 0.6, 0.8]
FIXED_CONF = 0.4  # used for experiments 1 and 2


# ---------- video + perturbations ----------
def load_frames(path, stride=5, max_frames=120, width=640):
    cap = cv2.VideoCapture(path)
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            h, w = f.shape[:2]
            f = cv2.resize(f, (width, int(h * width / w)))
            frames.append(f)
        i += 1
    cap.release()
    if not frames:
        raise SystemExit(f"Could not read frames from {path}")
    return frames


def darken(f):
    return cv2.convertScaleAbs(f, alpha=0.35, beta=0)


def blur(f):
    return cv2.GaussianBlur(f, (21, 21), 0)


def noise(f, sigma=40):
    n = np.random.normal(0, sigma, f.shape)
    return np.clip(f.astype(np.float32) + n, 0, 255).astype(np.uint8)


CONDITIONS = {
    "original": lambda f: f,
    "dark": darken,
    "blur": blur,
    "noise": noise,
}


# ---------- evaluation ----------
def evaluate(model, frames, conf):
    counts, confs = [], []
    for f in frames:
        r = model(f, conf=conf, verbose=False)[0]
        counts.append(len(r.boxes))
        confs.extend(r.boxes.conf.tolist())
    return float(np.mean(counts)), (float(np.mean(confs)) if confs else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--stride", type=int, default=5, help="use every Nth frame")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(0)

    base = load_frames(args.video, stride=args.stride)
    # Build each degraded version once so every model/conf sees identical frames
    versions = {name: [fn(f) for f in base] for name, fn in CONDITIONS.items()}

    # results[model][condition][conf] = (avg_dets, avg_conf)
    results = {}
    for m in MODELS:
        model = YOLO(m)
        results[m] = {}
        for cname, frames in versions.items():
            results[m][cname] = {}
            for c in CONFS:
                results[m][cname][c] = evaluate(model, frames, c)
                print(f"{m:12s} {cname:9s} conf={c}: "
                      f"dets/frame={results[m][cname][c][0]:.2f} "
                      f"mean_conf={results[m][cname][c][1]:.2f}")

    names = list(CONDITIONS)

    # ---- Chart 1: degradations (grouped by model) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(names))
    w = 0.38
    for k, m in enumerate(MODELS):
        axes[0].bar(x + (k - 0.5) * w,
                    [results[m][n][FIXED_CONF][0] for n in names], w,
                    label=m.replace(".pt", ""))
        axes[1].bar(x + (k - 0.5) * w,
                    [results[m][n][FIXED_CONF][1] for n in names], w,
                    label=m.replace(".pt", ""))
    for ax, title in zip(axes, ["Avg detections per frame", "Avg confidence"]):
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_title(f"{title} (conf={FIXED_CONF})")
        ax.legend()
    fig.suptitle("Experiments 1 & 2: harder video + model comparison")
    fig.tight_layout()
    fig.savefig(f"{args.out}/1_degradation_and_models.png", dpi=150)

    # ---- Chart 2: confidence sweep on original video ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m in MODELS:
        label = m.replace(".pt", "")
        axes[0].plot(CONFS, [results[m]["original"][c][0] for c in CONFS],
                     marker="o", label=label)
        axes[1].plot(CONFS, [results[m]["original"][c][1] for c in CONFS],
                     marker="o", label=label)
    axes[0].set_ylabel("Avg detections per frame")
    axes[1].set_ylabel("Avg confidence")
    for ax in axes:
        ax.set_xlabel("Confidence threshold")
        ax.set_xticks(CONFS)
        ax.legend()
    fig.suptitle("Experiment 3: confidence sweep (original video)")
    fig.tight_layout()
    fig.savefig(f"{args.out}/2_confidence_sweep.png", dpi=150)

    # ---- Chart 3: sweep x degradation, per model ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, m in zip(axes, MODELS):
        for n in names:
            ax.plot(CONFS, [results[m][n][c][0] for c in CONFS],
                    marker="o", label=n)
        ax.set_title(m.replace(".pt", ""))
        ax.set_xlabel("Confidence threshold")
        ax.set_xticks(CONFS)
        ax.legend()
    axes[0].set_ylabel("Avg detections per frame")
    fig.suptitle("Confidence sweep under each degradation")
    fig.tight_layout()
    fig.savefig(f"{args.out}/3_sweep_by_condition.png", dpi=150)

    print(f"\nSaved charts to {args.out}/")


if __name__ == "__main__":
    main()
