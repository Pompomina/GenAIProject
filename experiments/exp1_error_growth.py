"""Experiment 1: Error growth curves and power-law fitting (H1).

For each model × game:
  1. Load cached rollout (or fail with instructions to run rollouts first)
  2. Compute per-step MSE E_k
  3. Fit power law E_k ~ c * k^alpha; derive scale-invariant k*
  4. Save CSV and plots

Outputs:
  results/exp1/error_growth_{model}_{game}.csv
  results/exp1/alpha_kstar_table.csv
  results/exp1/error_curves_{game}.png
  results/exp1/error_curves_pixel_{game}.png  (pixel-space models only)
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import per_step_mse, fit_power_law
from src import visualize

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "experiment.yaml")
ROLLOUT_DIR = "results/rollouts"
OUT_DIR = "results/exp1"


def main():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    games = cfg["games"]
    K = cfg["horizon_K"]
    k_star_mult = cfg["k_star_multiplier"]
    model_cfgs = cfg["models"]

    os.makedirs(OUT_DIR, exist_ok=True)

    summary_rows = []
    # {model: {game: E_k}}
    all_error_curves: dict[str, dict[str, np.ndarray]] = {m: {} for m in model_cfgs}

    for model_name, mcfg in model_cfgs.items():
        space = mcfg["space"]
        for game in games:
            cache_path = os.path.join(ROLLOUT_DIR, f"{model_name}_{game}.npz")
            if not os.path.exists(cache_path):
                print(f"[exp1] WARNING: rollout cache missing for {model_name}/{game}: {cache_path}")
                print("  Run rollout.py first (or submit run_rollouts.sbatch).")
                continue

            data = np.load(cache_path)
            pred_frames = data["pred_frames"]
            true_frames = data["true_frames"]

            E_k = per_step_mse(pred_frames, true_frames)
            fit = fit_power_law(E_k, k_star_multiplier=k_star_mult)

            # Save per-model-game CSV
            df = pd.DataFrame({"k": np.arange(1, len(E_k) + 1), "mse": E_k})
            csv_path = os.path.join(OUT_DIR, f"error_growth_{model_name}_{game}.csv")
            df.to_csv(csv_path, index=False)

            all_error_curves[model_name][game] = E_k
            summary_rows.append({
                "model": model_name,
                "space": space,
                "game": game,
                "alpha": fit["alpha"],
                "c": fit["c"],
                "k_star": fit["k_star"],
                "fit_r2": fit["fit_r2"],
            })
            print(f"[exp1] {model_name}/{game}: alpha={fit['alpha']:.3f}, k*={fit['k_star']}, R2={fit['fit_r2']:.3f}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(OUT_DIR, "alpha_kstar_table.csv"), index=False)
        print("\n=== Alpha / k* table ===")
        print(summary_df.to_string(index=False))

        # Note: pixel-only alpha comparison is the valid scientific comparison.
        # MLP operates in RAM space and its alpha is NOT directly comparable.
        for game in games:
            if any(game in v for v in all_error_curves.values()):
                visualize.plot_error_curves(
                    all_error_curves, save_path=os.path.join(OUT_DIR, f"error_curves_{game}.png"))
                visualize.plot_error_curves(
                    all_error_curves, save_path=os.path.join(OUT_DIR, f"error_curves_pixel_{game}.png"),
                    pixel_only=True)
    else:
        print("[exp1] No rollout caches found. Run rollouts first.")


if __name__ == "__main__":
    main()
