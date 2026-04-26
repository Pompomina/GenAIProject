"""Thin wrapper around the official IRIS PyTorch checkpoint (ICLR 2023).

IRIS uses a VQ-VAE tokenizer + autoregressive Transformer to predict next frames.
Checkpoint must be downloaded separately; see scripts/download_checkpoints.sh.
"""
from __future__ import annotations
import numpy as np
import torch


class IRISWrapper:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.checkpoint_path = checkpoint_path

    def load_checkpoint(self) -> None:
        # Import here so missing IRIS install only errors if this class is actually used
        try:
            from iris.agent import Agent  # type: ignore[import]
        except ImportError as e:
            raise ImportError("IRIS not installed. Run scripts/setup_env_torch.sh") from e

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model = Agent.load_from_checkpoint(checkpoint).to(self.device)
        self.model.eval()

    def predict(self, obs: np.ndarray, action: int) -> np.ndarray:
        """Single-step autoregressive prediction.

        obs: uint8 [84, 84, 3] — the model's current predicted frame (not the real frame
             after step 0; caller is responsible for passing chained predictions).
        action: scalar int
        Returns: uint8 [84, 84, 3] predicted next frame.
        """
        assert self.model is not None, "Call load_checkpoint() first"
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            obs_t = obs_t.to(self.device)
            act_t = torch.tensor([action], dtype=torch.long).to(self.device)
            next_obs = self.model.world_model.predict_next_obs(obs_t, act_t)
            next_obs = (next_obs.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return next_obs

    def get_actor(self):
        """Return the trained actor for collecting action sequences."""
        assert self.model is not None, "Call load_checkpoint() first"
        return self.model.actor
