"""Stage 2: build the paired dataset from demo and rollout states. Resumable per task."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

SUITES = ("libero_spatial", "libero_object")
MAGNITUDES = (0.2, 0.4, 0.6, 0.8, 1.0)
DISTRACTOR_MAGNITUDES = (0.25, 0.5, 0.75, 1.0)
DEMO_ROOT = Path("/cache/libero_datasets")
LIFT_THRESHOLD = 0.02
GRASP_DISTANCE = 0.08


def magnitudes_for(axis: str) -> tuple[float, ...]:
    return DISTRACTOR_MAGNITUDES if axis == "distractors" else MAGNITUDES


def demo_samples(path: Path, count: int, rng: np.random.Generator) -> list[dict]:
    import h5py

    samples = []
    with h5py.File(path, "r") as handle:
        demos = sorted(handle["data"], key=lambda name: int(name.split("_")[1]))
        for index in rng.choice(len(demos), size=count, replace=False):
            group = handle["data"][demos[index]]
            states = group["states"]
            frame = int(rng.integers(len(states)))
            samples.append(
                {
                    "state": states[frame],
                    "initial": states[0],
                    "action": group["actions"][frame].astype(np.float32),
                    "source": "demo",
                }
            )
    return samples


def rollout_samples(path: Path, count: int, rng: np.random.Generator) -> list[dict]:
    samples = []
    with np.load(path) as data:
        states = data["states_0"]
    for frame in sorted(rng.choice(len(states), size=count, replace=False).tolist()):
        samples.append(
            {
                "state": states[frame],
                "initial": states[0],
                "action": np.full(7, np.nan, dtype=np.float32),
                "source": "rollout",
            }
        )
    return samples


def task_phase(eef: np.ndarray, target: np.ndarray, initial_target: np.ndarray) -> str:
    if target[2] - initial_target[2] > LIFT_THRESHOLD:
        return "transport"
    if np.linalg.norm(eef - target) < GRASP_DISTANCE:
        return "grasp"
    return "approach"


def build_task(
    suite: str,
    task_id: int,
    demo_count: int,
    rollout_count: int,
    seed: int,
    rollouts: Path,
    output: Path,
) -> str:
    from faultline_vla.envs.paired import make_scene, render_pair, set_sim_state
    from faultline_vla.envs.perturb import AXES
    from faultline_vla.models.loaders import proprio_from_obs

    if output.exists():
        return f"skip {output.name}"
    rng = np.random.default_rng([seed, SUITES.index(suite), task_id])
    scene = make_scene(suite, task_id)
    from libero.libero import benchmark

    task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
    samples = demo_samples(DEMO_ROOT / suite / f"{task.name}_demo.hdf5", demo_count, rng)
    samples += rollout_samples(rollouts / f"{suite}_task{task_id}.npz", rollout_count, rng)
    target = scene.env.env.obj_of_interest[0]

    def observe(state: np.ndarray) -> dict:
        set_sim_state(scene, state)
        return scene.env.env._get_observations(force_update=True)

    columns: dict[str, list] = {
        name: [] for name in ("states", "proprio", "clean", "wrist", "demo_action", "source", "phase")
    }
    perturbed: dict[str, list] = {}
    for index, sample in enumerate(samples):
        initial_target = observe(sample["initial"])[f"{target}_pos"]
        raw = observe(sample["state"])
        columns["states"].append(sample["state"])
        columns["proprio"].append(proprio_from_obs(raw))
        columns["demo_action"].append(sample["action"])
        columns["source"].append(sample["source"])
        columns["phase"].append(
            task_phase(raw["robot0_eef_pos"], raw[f"{target}_pos"], initial_target)
        )
        for axis in AXES:
            for m in magnitudes_for(axis):
                pair = render_pair(scene, sample["state"], axis, m, seed=seed + index, key=index)
                perturbed.setdefault(f"{axis}/{m}", []).append(pair["perturbed"])
        columns["clean"].append(pair["clean"])
        columns["wrist"].append(pair["wrist"])
    arrays = {name: np.stack(values) for name, values in columns.items()}
    arrays.update({f"perturbed/{key}": np.stack(value) for key, value in perturbed.items()})
    arrays["language"] = np.array(scene.language)
    scene.env.close()
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.rename(output)
    return f"wrote {output.name}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--demo-states", type=int, default=2)
    parser.add_argument("--rollout-states", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rollouts", type=Path, default=Path("data/stage2/rollouts"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/stage2"))
    args = parser.parse_args()

    config = {
        "tasks": args.tasks,
        "demo_states": args.demo_states,
        "rollout_states": args.rollout_states,
        "seed": args.seed,
        "magnitudes": MAGNITUDES,
        "distractor_magnitudes": DISTRACTOR_MAGNITUDES,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output_dir = args.output_dir / config_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    jobs = [
        (
            suite,
            task,
            args.demo_states,
            args.rollout_states,
            args.seed,
            args.rollouts,
            output_dir / f"{suite}_task{task}.npz",
        )
        for suite in SUITES
        for task in range(args.tasks)
    ]
    print(f"dataset {config_hash}: {len(jobs)} tasks", flush=True)
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for message in pool.starmap(build_task, jobs, chunksize=1):
            print(message, flush=True)


if __name__ == "__main__":
    main()
