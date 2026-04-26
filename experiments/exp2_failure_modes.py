"""Experiment 2: Architecture-specific failure mode visualization (H2).

For each model × game, samples representative trajectories and renders frames
at k=1, 10, 50 side-by-side for visual inspection of how each model degrades.

Expected observations:
  IRIS: blur and spectral artifacts (VQ-VAE quantization residuals compound)
  DIAMOND: maintains visual coherence longer (diffusion denoising resists drift)
  DreamerV3: geometrically plausible but structurally incorrect (GRU drift)
  MLP: degrades rapidly (RAM space — different failure character than pixel models)

Outputs:
  results/exp2/frames_{model}_{game}.png     (per-model frame strip)
  results/exp2/comparison_{game}.png         (all models side-by-side)
"""
import os
import sys
import yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import visualize

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "experiment.yaml")
ROLLOUT_DIR = "results/rollouts"
OUT_DIR = "results/exp2"

VIS_K = [0, 9, 49]  # 0-indexed: k=1, 10, 50
N_SAMPLE_TRAJ = 5   # number of representative trajectories to visualize


def main():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    games = cfg["games"]
    model_cfgs = cfg["models"]
    os.makedirs(OUT_DIR, exist_ok=True)

    for game in games:
        frames_by_model: dict[str, np.ndarray] = {}
        true_frames_by_model: dict[str, np.ndarray] = {}

        for model_name in model_cfgs:
            cache_path = os.path.join(ROLLOUT_DIR, f"{model_name}_{game}.npz")
            if not os.path.exists(cache_path):
                print(f"[exp2] WARNING: rollout cache missing for {model_name}/{game}")
                continue

            data = np.load(cache_path)
            pred_frames = data["pred_frames"]   # [n_traj, K, ...]
            true_frames = data["true_frames"]

            # Pick N_SAMPLE_TRAJ evenly spaced trajectories
            idxs = np.linspace(0, len(pred_frames) - 1, N_SAMPLE_TRAJ, dtype=int)

            for i, traj_idx in enumerate(idxs):
                traj_pred = pred_frames[traj_idx]   # [K, ...]
                traj_true = true_frames[traj_idx]

                # For pixel models, show frames; for MLP (RAM), skip frame grid
                space = model_cfgs[model_name]["space"]
                if space != "pixel":
                    continue

                frames_by_model[model_name] = traj_pred
                true_frames_by_model[model_name] = traj_true

        if not frames_by_model:
            print(f"[exp2] No pixel-space rollout caches found for {game}. Skipping.")
            continue

        # Combined comparison: all pixel models side-by-side with ground truth
        visualize.plot_frame_grid(
            frames_by_model=frames_by_model,
            ks=VIS_K,
            game=game,
            true_frames=true_frames_by_model,
            save_path=os.path.join(OUT_DIR, f"comparison_{game}.png"),
        )

        # Per-model strip (no ground truth, just the model's output)
        for model_name, traj_frames in frames_by_model.items():
            visualize.plot_frame_grid(
                frames_by_model={model_name: traj_frames},
                ks=VIS_K,
                game=game,
                save_path=os.path.join(OUT_DIR, f"frames_{model_name}_{game}.png"),
            )


if __name__ == "__main__":
    main()
