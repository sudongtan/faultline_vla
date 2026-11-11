"""Tests that perturbations change pixels, are reproducible and leave physics alone."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest

from faultline_vla.envs.paired import (
    IMAGE_SIZE,
    WRIST,
    make_scene,
    render,
    render_pair,
    set_sim_state,
)
from faultline_vla.envs.perturb import AGENTVIEW, AXES, SCENE_AXES


@pytest.fixture(scope="module", params=["libero_spatial", "libero_object"])
def scene(request):
    built = make_scene(request.param, 0)
    yield built
    built.env.close()


def state_of(scene) -> np.ndarray:
    return scene.init_states[0]


def test_magnitude_zero_is_clean(scene):
    state = state_of(scene)
    for axis in AXES:
        pair = render_pair(scene, state, axis, 0.0, seed=1)
        assert np.array_equal(pair["clean"], pair["perturbed"]), axis


@pytest.mark.parametrize("axis", AXES)
def test_full_magnitude_changes_agentview(scene, axis):
    pair = render_pair(scene, state_of(scene), axis, 1.0, seed=1)
    assert np.abs(pair["clean"].astype(float) - pair["perturbed"].astype(float)).mean() > 0.5


@pytest.mark.parametrize("axis", AXES)
def test_same_seed_is_reproducible(scene, axis):
    state = state_of(scene)
    first = render_pair(scene, state, axis, 0.75, seed=3)
    second = render_pair(scene, state, axis, 0.75, seed=3)
    assert np.array_equal(first["perturbed"], second["perturbed"])


@pytest.mark.parametrize("axis", AXES)
def test_scene_is_restored_after_perturbing(scene, axis):
    state = state_of(scene)
    set_sim_state(scene, state)
    before = render(scene, AGENTVIEW)
    render_pair(scene, state, axis, 1.0, seed=2)
    set_sim_state(scene, state)
    assert np.array_equal(before, render(scene, AGENTVIEW))


@pytest.mark.parametrize("axis", AXES)
def test_physics_state_is_unchanged(scene, axis):
    state = state_of(scene)
    set_sim_state(scene, state)
    render_pair(scene, state, axis, 1.0, seed=2)
    after = scene.env.env.sim.get_state().flatten()
    assert np.array_equal(after, state)


def test_wrist_camera_is_not_perturbed(scene):
    state = state_of(scene)
    baseline = render_pair(scene, state, "light_dim", 0.0, seed=1)["wrist"]
    assert np.array_equal(baseline, render_pair(scene, state, "camera_pose", 1.0, seed=1)["wrist"])


def test_render_size(scene):
    assert render(scene, AGENTVIEW).shape == (IMAGE_SIZE, IMAGE_SIZE, 3)


@pytest.mark.parametrize("axis", SCENE_AXES)
def test_snapshot_restores_the_same_perturbation(scene, axis):
    set_sim_state(scene, state_of(scene))
    scene.perturber.apply(axis, 1.0, 0)
    expected = render(scene, AGENTVIEW)
    held = scene.perturber.snapshot()
    scene.perturber.reset()
    scene.perturber.restore(held)
    restored = render(scene, AGENTVIEW)
    scene.perturber.reset()
    assert np.array_equal(restored, expected)


def _render_unmodified(suite: str, task_id: int, state: np.ndarray) -> list[np.ndarray]:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=IMAGE_SIZE, camera_widths=IMAGE_SIZE
    )
    env.seed(0)
    env.reset()
    env.env.sim.set_state_from_flattened(state)
    env.env.sim.forward()
    images = [
        env.env.sim.render(camera_name=c, height=IMAGE_SIZE, width=IMAGE_SIZE)[::-1, ::-1].copy()
        for c in (AGENTVIEW, WRIST)
    ]
    env.close()
    return images


def test_parked_distractors_leave_the_scene_unchanged(scene):
    # A second OSMesa context in one process renders wrongly, so the reference gets its own process.
    with mp.get_context("spawn").Pool(1) as pool:
        reference = pool.apply(_render_unmodified, (scene.suite, scene.task_id, state_of(scene)))
    set_sim_state(scene, state_of(scene))
    scene.perturber.reset()
    for camera, expected in zip((AGENTVIEW, WRIST), reference, strict=True):
        assert np.array_equal(render(scene, camera), expected)
