"""Stage 2b: does any visual perturbation reduce closed-loop task success?

Follows the Stage 2 open-loop gate. Runs paired episodes (same init state, env seed and policy
noise) under clean and perturbed views. Only the agentview camera is perturbed, as in Stage 2.
Resumable: every episode is written to its own JSON file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from faultline_vla.jobs import run_jobs

SUITES = ("libero_spatial", "libero_object")
MAX_STEPS = 280
SETTLE_STEPS = 10
NOOP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
CLEAN = ("clean", 0.0)
# Chosen from Stage 2 before any closed-loop data: the four axes with the largest open-loop change
# relative to the noise floor, plus the smallest as a negative control.
CONDITIONS = (
    CLEAN,
    ("sensor_noise", 1.0),
    ("camera_pose", 1.0),
    ("camera_fov", 1.0),
    ("distractors", 1.0),
    ("light_bright", 1.0),
)
SECONDS_PER_EPISODE = 65.0
ENV_SEED = 3000
NOISE_SEED = 4000


def observe(scene, axis: str, m: float, held, perturb_seed: int, key: int) -> tuple[np.ndarray, np.ndarray]:
    """Render a perturbed agentview and an unperturbed wrist view of the current state."""
    from faultline_vla.envs.paired import WRIST, render
    from faultline_vla.envs.perturb import AGENTVIEW, apply_image_axis

    perturber = scene.perturber
    perturber.reset()
    wrist = render(scene, WRIST)
    if axis == "clean":
        return render(scene, AGENTVIEW), wrist
    perturber.restore(held)
    agentview = apply_image_axis(render(scene, AGENTVIEW), axis, m, perturb_seed, key)
    perturber.reset()
    return agentview, wrist


def run_episode(runner, suite: str, task_id: int, episode: int, axis: str, m: float, action_steps: int) -> dict:
    from faultline_vla.envs.paired import make_scene
    from faultline_vla.models.loaders import proprio_from_obs

    # robosuite's hard reset rebuilds the simulator and would drop the distractor bodies, so each
    # episode starts from a newly built scene instead of env.reset().
    scene = make_scene(suite, task_id)
    env = scene.env
    env_seed = ENV_SEED + 1000 * task_id + episode
    noise_seed = NOISE_SEED + 1000 * task_id + episode
    perturb_seed = 1000 * task_id + episode
    env.seed(env_seed)
    raw = env.set_init_state(scene.init_states[episode])
    for _ in range(SETTLE_STEPS):
        raw, _, _, _ = env.step(NOOP)
    for robot in env.robots:
        robot.controller.use_delta = True

    # Draw the perturbation once. Distractor placement depends on current object positions, so
    # re-drawing it at each replan would move the distractors during the episode.
    held = None
    if axis != "clean":
        scene.perturber.apply(axis, m, perturb_seed)
        held = scene.perturber.snapshot()
        scene.perturber.reset()

    queue: list[np.ndarray] = []
    success, calls, first_view = False, 0, None
    step = 0
    for step in range(MAX_STEPS):
        if not queue:
            agentview, wrist = observe(scene, axis, m, held, perturb_seed, step)
            first_view = agentview if first_view is None else first_view
            chunk = runner.predict(agentview, wrist, proprio_from_obs(raw), scene.language, runner.make_noise(noise_seed + step))
            queue = list(chunk[:action_steps])
            calls += 1
        raw, _, done, _ = env.step(queue.pop(0))
        if env.check_success():
            success = True
            break
        if done:
            break
    env.close()
    return {
        "suite": suite, "task_id": task_id, "episode": episode, "axis": axis, "m": m,
        "success": success, "steps": step + 1, "calls": calls,
        "env_seed": env_seed, "noise_seed": noise_seed, "perturb_seed": perturb_seed,
        "first_view": first_view,
    }


def run_condition(suite: str, task_id: int, episode: int, axis: str, m: float, action_steps: int, path: Path, threads: int, device: str) -> None:
    """Run one episode and save it. Runs in its own process, so memory is returned after each one."""
    # Load torch before creating the OSMesa context, the same order LeRobot's evaluator uses.
    from faultline_vla.models.loaders import PolicyRunner

    runner = PolicyRunner(device, threads)
    start = time.perf_counter()
    result = run_episode(runner, suite, task_id, episode, axis, m, action_steps)
    view = result.pop("first_view")
    result["seconds"] = round(time.perf_counter() - start, 1)
    np.savez_compressed(path.with_suffix(".npz"), first_view=view)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result))
    temporary.rename(path)
    print(json.dumps(result), flush=True)


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def paired_drop(frame: pd.DataFrame, axis: str, resamples: int, seed: int = 0) -> dict[str, float]:
    """Clean minus perturbed success, with a CI from resampling tasks. Only complete pairs count."""
    key = ["task_id", "episode"]
    pair = frame[frame.axis == "clean"].merge(frame[frame.axis == axis], on=key, suffixes=("_clean", "_pert"))
    difference = pair.success_clean.astype(float) - pair.success_pert.astype(float)
    tasks = difference.groupby(pair.task_id).mean().to_numpy()
    if len(tasks) == 0:
        return {"pairs": 0, "drop": float("nan"), "drop_low": float("nan"), "drop_high": float("nan")}
    samples = np.random.default_rng(seed).choice(tasks, size=(resamples, len(tasks))).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"pairs": len(pair), "drop": float(difference.mean()), "drop_low": float(low), "drop_high": float(high)}


def analyze(output: Path, results: Path, tag: str, conditions: list, expected: int, resamples: int) -> None:
    import matplotlib.pyplot as plt

    from faultline_vla.viz import AXIS_LABELS, SUITE_COLORS, SUITE_LABELS, apply_style

    frame = pd.DataFrame([json.loads(path.read_text()) for path in sorted(output.glob("*.json"))])
    frame["config_hash"] = tag
    (results / "tables").mkdir(parents=True, exist_ok=True)
    frame.to_csv(results / "tables" / f"stage2b_episodes_{tag}.csv", index=False)
    rows = []
    for suite in SUITES:
        data = frame[frame.suite == suite]
        for axis, _ in conditions:
            group = data[data.axis == axis]
            low, high = wilson(int(group.success.sum()), len(group))
            row = {"suite": suite, "axis": axis, "n": len(group), "expected": expected, "successes": int(group.success.sum()), "rate": group.success.mean(), "low": low, "high": high}
            if axis != "clean":
                row |= paired_drop(data, axis, resamples)
                row["success_drop"] = bool(row["drop_low"] > 0)
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(results / "tables" / f"stage2b_summary_{tag}.csv", index=False)
    print(summary.round(3).to_string(), flush=True)

    apply_style()
    labels = {"clean": "Clean", **AXIS_LABELS}
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)
    order = [axis for axis, _ in conditions]
    for plot, (suite, color) in zip(axes, SUITE_COLORS.items(), strict=True):
        data = summary[summary.suite == suite].set_index("axis").loc[order]
        positions = np.arange(len(order))
        errors = [data.rate - data.low, data.high - data.rate]
        plot.errorbar(data.rate, positions, xerr=errors, fmt="o", ms=6, color=color, lw=2)
        plot.set_yticks(positions, [labels[axis] for axis in order])
        plot.set_title(SUITE_LABELS[suite], loc="left")
        plot.set_xlabel("Task success (Wilson 95% CI)")
        plot.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    axes[0].set_xlim(-0.03, 1.03)
    figure.tight_layout()
    (results / "figs").mkdir(parents=True, exist_ok=True)
    figure.savefig(results / "figs" / "stage2b_success.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--episode-limit", type=float, default=900.0, help="Seconds before an episode is killed and retried.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    config = {"tasks": args.tasks, "episodes": args.episodes, "action_steps": args.action_steps, "conditions": CONDITIONS, "max_steps": MAX_STEPS, "env_seed": ENV_SEED, "noise_seed": NOISE_SEED}
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = args.results_dir / "stage2b" / tag
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.txt").write_text(json.dumps(config, indent=2))
    expected = args.tasks * args.episodes
    conditions = list(CONDITIONS)
    if not args.analyze_only:
        # Episode index is the outer loop, so a partial run stays balanced across tasks and conditions.
        jobs = []
        for episode in range(args.episodes):
            for suite in SUITES:
                for task in range(args.tasks):
                    for axis, m in conditions:
                        path = output / f"{suite}_task{task}_ep{episode}_{axis}_{m}.json"
                        if not path.exists():
                            jobs.append((suite, task, episode, axis, m, args.action_steps, path, args.threads, args.device))
        print(f"tag {tag}: {len(jobs)} episodes to run, estimated {len(jobs) * SECONDS_PER_EPISODE / 3600:.1f} h", flush=True)
        start = time.perf_counter()
        failed = run_jobs(run_condition, jobs, lambda job: job[6], args.workers, args.episode_limit, args.retries)
        print(f"episodes took {(time.perf_counter() - start) / 3600:.2f} h; {len(failed)} never finished", flush=True)
    analyze(output, args.results_dir, tag, conditions, expected, args.resamples)


if __name__ == "__main__":
    main()
