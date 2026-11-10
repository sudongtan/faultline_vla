"""Load SmolVLA and feed it observations the same way LeRobot's evaluator does."""

from __future__ import annotations

import numpy as np
import torch

CHECKPOINT = "HuggingFaceVLA/smolvla_libero"
ACTION_DIM = 7
AGENTVIEW_KEY = "observation.images.image"
WRIST_KEY = "observation.images.image2"
STATE_KEY = "observation.state"


def quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion to axis-angle, matching LeRobot's LIBERO integration."""
    quaternion = np.asarray(quaternion, dtype=np.float32)
    w = np.clip(quaternion[..., 3], -1, 1)
    denominator = np.sqrt(np.maximum(1 - w * w, 0))
    scale = np.divide(
        2 * np.arccos(w), denominator, out=np.zeros_like(denominator), where=denominator > 1e-10
    )
    return (quaternion[..., :3] * scale[..., None]).astype(np.float32)


def proprio_from_obs(raw: dict) -> np.ndarray:
    """Build the 8-D state from a raw LIBERO observation."""
    return np.concatenate(
        (raw["robot0_eef_pos"], quat_to_axis_angle(raw["robot0_eef_quat"]), raw["robot0_gripper_qpos"])
    ).astype(np.float32)


def noise_for(seed: int, chunk_size: int, action_dim: int, batch: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, chunk_size, action_dim, generator=generator)


class PolicyRunner:
    """SmolVLA in float32 with the checkpoint's own pre- and post-processing."""

    def __init__(self, device: str = "cpu", threads: int = 6, checkpoint: str = CHECKPOINT):
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        torch.set_num_threads(threads)
        if device == "cpu":
            torch.backends.mkldnn.enabled = False
        self.device = device
        policy = SmolVLAPolicy.from_pretrained(checkpoint)
        policy.config.device = device
        policy.to(device)
        policy.eval()
        policy.float()
        self.policy = policy
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": device}},
        )

    @property
    def chunk_size(self) -> int:
        return self.policy.config.chunk_size

    @property
    def max_action_dim(self) -> int:
        return self.policy.config.max_action_dim

    def make_noise(self, seed: int) -> torch.Tensor:
        return noise_for(seed, self.chunk_size, self.max_action_dim).to(self.device)

    def prepare(
        self, agentview: np.ndarray, wrist: np.ndarray, proprio: np.ndarray, language: str
    ) -> dict:
        """Turn uint8 HWC images, an 8-D state and an instruction into a model batch."""

        def image_tensor(image: np.ndarray) -> torch.Tensor:
            array = np.ascontiguousarray(image)
            return (torch.from_numpy(array).permute(2, 0, 1)[None].float() / 255.0).to(self.device)

        return self.preprocessor(
            {
                AGENTVIEW_KEY: image_tensor(agentview),
                WRIST_KEY: image_tensor(wrist),
                STATE_KEY: torch.from_numpy(proprio)[None].float().to(self.device),
                "task": [language],
            }
        )

    @torch.inference_mode()
    def chunk(self, batch: dict, noise: torch.Tensor) -> np.ndarray:
        """Return the un-normalized action chunk with shape (chunk, 7)."""
        normalized = self.policy._get_action_chunk(batch, noise=noise)
        return self.postprocessor(normalized).cpu().numpy()[0]

    def predict(
        self,
        agentview: np.ndarray,
        wrist: np.ndarray,
        proprio: np.ndarray,
        language: str,
        noise: torch.Tensor,
    ) -> np.ndarray:
        return self.chunk(self.prepare(agentview, wrist, proprio, language), noise)
