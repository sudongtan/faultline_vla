"""Build LIBERO scenes and render the same simulator state clean and perturbed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from faultline_vla.envs.perturb import (
    AGENTVIEW,
    Perturber,
    add_distractor_bodies,
    apply_image_axis,
)

WRIST = "robot0_eye_in_hand"
IMAGE_SIZE = 256


@dataclass
class Scene:
    env: object
    perturber: Perturber
    suite: str
    task_id: int
    language: str
    init_states: np.ndarray


def make_scene(suite_name: str, task_id: int, image_size: int = IMAGE_SIZE) -> Scene:
    """Create a LIBERO task with parked distractors and a perturber."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=image_size, camera_widths=image_size
    )
    env.seed(0)
    env.reset()
    offsets = add_distractor_bodies(env)
    init_states = suite.get_task_init_states(task_id)
    env.set_init_state(init_states[0])
    textures_dir = Path(get_libero_path("assets")) / "textures"
    return Scene(
        env=env,
        perturber=Perturber(env, textures_dir, offsets),
        suite=suite_name,
        task_id=task_id,
        language=task.language,
        init_states=init_states,
    )


def set_sim_state(scene: Scene, state: np.ndarray) -> None:
    """Set the exact simulator state without stepping physics."""
    sim = scene.env.env.sim
    sim.set_state_from_flattened(state)
    sim.forward()


def render(scene: Scene, camera: str, size: int = IMAGE_SIZE) -> np.ndarray:
    """Render one camera the way LeRobot presents it to the policy."""
    image = scene.env.env.sim.render(camera_name=camera, height=size, width=size)
    return image[::-1, ::-1].copy()


def render_pair(
    scene: Scene, state: np.ndarray, axis: str, m: float, seed: int, key: int = 0
) -> dict[str, np.ndarray]:
    """Render ``state`` clean and perturbed. Only the agentview camera is perturbed."""
    set_sim_state(scene, state)
    scene.perturber.reset()
    clean = render(scene, AGENTVIEW)
    wrist = render(scene, WRIST)
    scene.perturber.apply(axis, m, seed)
    perturbed = apply_image_axis(render(scene, AGENTVIEW), axis, m, seed, key)
    scene.perturber.reset()
    return {"clean": clean, "perturbed": perturbed, "wrist": wrist}
