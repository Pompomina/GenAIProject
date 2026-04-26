"""Central rollout engine with fixed-action replay and result caching.

Fixed-action protocol:
- Actions are collected once from the IRIS/DIAMOND trained actor (not random — random
  actions push models OOD from their training distribution, inflating error equally across
  all architectures and obscuring architectural differences).
- Both the model and the real Atari env are stepped with identical action sequences from
  identical starting states — this controls the branching point.
- After step 0, models are stepped AUTOREGRESSIVELY on their own predictions.
  They never see real frames again after the initial observation.
  true_frames come from the real env; pred_frames come from chained model output.

Caching:
- results/rollouts/{model}_{game}.npz is expensive to recompute; never delete.
- rollout() skips computation if the cache file already exists.
"""
from __future__ import annotations
import os
from typing import Any
import numpy as np
import gymnasium as gym
import yaml


def _load_config() -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "experiment.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def collect_actions(
    game: str,
    n_trajectories: int,
    K: int,
    actor,
    seed: int = 42,
    save_path: str | None = None,
    device: str = "cuda",
) -> np.ndarray:
    """Collect action sequences using the trained actor from IRIS or DIAMOND.

    actor: object with .act(obs) -> int  (the model's trained policy)
    Returns: actions np.ndarray [n_trajectories, K] int32
    Saves to save_path if provided.
    """
    import torch
    env = gym.make(f"ALE/{game}-v5", obs_type="rgb", render_mode=None)
    rng = np.random.default_rng(seed)
    actions = np.zeros((n_trajectories, K), dtype=np.int32)

    for traj_idx in range(n_trajectories):
        seed_i = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=seed_i)
        for k in range(K):
            with torch.no_grad():
                action = actor.act(obs)
            actions[traj_idx, k] = action
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))

    env.close()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, actions=actions)
    return actions


def rollout(
    model_name: str,
    model: Any,
    game: str,
    n_trajectories: int,
    K: int,
    seed: int = 42,
    actions_path: str | None = None,
    cache_dir: str = "results/rollouts",
    obs_type: str = "rgb",
) -> dict:
    """Run fixed-action rollout for a given model and game, with caching.

    model_name: one of 'mlp', 'iris', 'diamond', 'dreamerv3'
    model: wrapper instance with .predict(obs, action) -> next_obs
           For dreamerv3, model is a DreamerV3Wrapper; rollout is handled differently.
    obs_type: 'rgb' for pixel models, 'ram' for MLP

    Returns dict with keys:
        pred_frames: np.ndarray [n_trajectories, K, ...obs_shape]
        true_frames: np.ndarray [n_trajectories, K, ...obs_shape]
        actions:     np.ndarray [n_trajectories, K]
    """
    cache_path = os.path.join(cache_dir, f"{model_name}_{game}.npz")
    if os.path.exists(cache_path):
        print(f"[rollout] Loading cached results from {cache_path}")
        data = np.load(cache_path)
        return {"pred_frames": data["pred_frames"], "true_frames": data["true_frames"], "actions": data["actions"]}

    # Load or collect actions
    if actions_path and os.path.exists(actions_path):
        actions = np.load(actions_path)["actions"]
    else:
        raise FileNotFoundError(
            f"Action file not found at {actions_path}. "
            "Collect actions first using collect_actions() with the IRIS or DIAMOND actor."
        )

    assert actions.shape[0] >= n_trajectories and actions.shape[1] >= K, (
        f"Action file shape {actions.shape} insufficient for n_traj={n_trajectories}, K={K}"
    )
    actions = actions[:n_trajectories, :K]

    # DreamerV3: dispatch to subprocess (env_jax)
    if model_name == "dreamerv3":
        output_path = cache_path
        pred_frames = model.run_rollout(
            game=game, actions_path=actions_path, output_path=output_path,
            n_traj=n_trajectories, K=K, seed=seed,
        )
        true_frames = _collect_true_frames(game, actions, n_trajectories, K, seed, obs_type)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache_path, pred_frames=pred_frames, true_frames=true_frames, actions=actions)
        return {"pred_frames": pred_frames, "true_frames": true_frames, "actions": actions}

    # All other models: PyTorch, step autoregressively in this process
    obs_shape = (128,) if obs_type == "ram" else (84, 84, 3)
    pred_frames = np.zeros((n_trajectories, K, *obs_shape), dtype=np.uint8)
    true_frames = np.zeros((n_trajectories, K, *obs_shape), dtype=np.uint8)

    env_obs_type = "ram" if obs_type == "ram" else "rgb"
    env = gym.make(f"ALE/{game}-v5", obs_type=env_obs_type, render_mode=None)
    rng = np.random.default_rng(seed)

    for traj_idx in range(n_trajectories):
        seed_i = int(rng.integers(0, 2**31))
        real_obs, _ = env.reset(seed=seed_i)
        model_obs = real_obs.copy()  # model starts from the same initial observation

        for k in range(K):
            action = int(actions[traj_idx, k])

            # Step real environment to get ground-truth next frame
            real_obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                real_obs, _ = env.reset()

            # Step model autoregressively on its own previous prediction (never real frames)
            model_obs = model.predict(model_obs, action)

            true_frames[traj_idx, k] = real_obs
            pred_frames[traj_idx, k] = model_obs

    env.close()

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_path, pred_frames=pred_frames, true_frames=true_frames, actions=actions)
    print(f"[rollout] Saved {model_name}_{game} rollout to {cache_path}")
    return {"pred_frames": pred_frames, "true_frames": true_frames, "actions": actions}


def _collect_true_frames(
    game: str,
    actions: np.ndarray,
    n_trajectories: int,
    K: int,
    seed: int,
    obs_type: str,
) -> np.ndarray:
    """Collect ground-truth frames by replaying the same action sequences in the real env."""
    obs_shape = (128,) if obs_type == "ram" else (84, 84, 3)
    true_frames = np.zeros((n_trajectories, K, *obs_shape), dtype=np.uint8)
    env_obs_type = "ram" if obs_type == "ram" else "rgb"
    env = gym.make(f"ALE/{game}-v5", obs_type=env_obs_type, render_mode=None)
    rng = np.random.default_rng(seed)

    for traj_idx in range(n_trajectories):
        seed_i = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=seed_i)
        for k in range(K):
            action = int(actions[traj_idx, k])
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
            true_frames[traj_idx, k] = obs

    env.close()
    return true_frames
