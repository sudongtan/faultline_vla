"""Stage 3 probes: does a perturbation destroy task geometry in the features, or only move it?

Ridge probes read the target object's world position and the gripper-to-target vector from agentview
features at several depths. Each probe is trained on clean features and tested on perturbed ones
(transfer), and also refit on perturbed features (refit). Tasks are held out. High refit with low
transfer means the information is still there in a different form; both low means it is lost.
Decodable means present, not used by the policy (Belinkov 2022). Resumable per task.
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

PROBE_SEED = 5000
GRID = 4
PIXEL_GRID = 16
FOLDS = 5
ALPHAS = np.logspace(-1, 6, 15)
TARGETS = ("target_pos", "eef_to_target")
SECONDS_PER_STATE = 30  # Measured at 3 threads while other runs shared the CPU.


def pool(tokens, grid: int) -> np.ndarray:
    """Average a square token grid down to grid × grid cells and flatten."""
    count, dim = tokens.shape
    side = round(count**0.5)
    assert side * side == count and side % grid == 0, tokens.shape
    cells = tokens.float().reshape(grid, side // grid, grid, side // grid, dim).mean(dim=(1, 3))
    return cells.reshape(-1).numpy().astype(np.float16)


def pool_pixels(image: np.ndarray) -> np.ndarray:
    side = image.shape[0] // PIXEL_GRID
    cells = image.reshape(PIXEL_GRID, side, PIXEL_GRID, side, 3).mean(axis=(1, 3))
    return (cells.reshape(-1) / 255.0).astype(np.float16)


def collect_task(
    suite: str,
    task_id: int,
    conditions: list[str],
    states: int,
    output: Path,
    threads: int,
    device: str,
) -> str:
    from faultline_vla.models.loaders import PolicyRunner, proprio_from_obs
    from faultline_vla.models.patching import ReadoutPatcher

    if output.exists():
        return f"skip {output.name}"
    # Load the policy before importing LIBERO. The other order segfaults while torch loads triton.
    runner = PolicyRunner(device, threads)
    patcher = ReadoutPatcher(runner)
    from libero.libero import benchmark
    from stage2_build_dataset import DEMO_ROOT, SUITES, demo_samples

    from faultline_vla.envs.paired import make_scene, render_pair, set_sim_state

    vision_count = len(patcher.backbone.get_vlm_model().vision_model.encoder.layers)
    vision_blocks = list(range(2, vision_count, 3))
    layers = list(range(0, patcher.num_layers, 4)) + [patcher.num_layers - 1]
    sites = (
        [f"vision_{b:02d}" for b in vision_blocks]
        + ["connector"]
        + [f"vlm_{n:02d}" for n in layers]
    )

    rng = np.random.default_rng([PROBE_SEED, SUITES.index(suite), task_id])
    scene = make_scene(suite, task_id)
    task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
    samples = demo_samples(DEMO_ROOT / suite / f"{task.name}_demo.hdf5", states, rng)
    target = scene.env.env.obj_of_interest[0]

    labels: dict[str, list] = {name: [] for name in TARGETS}
    features: dict[str, list] = {}
    for index, sample in enumerate(samples):
        set_sim_state(scene, sample["state"])
        raw = scene.env.env._get_observations(force_update=True)
        labels["target_pos"].append(raw[f"{target}_pos"])
        labels["eef_to_target"].append(raw[f"{target}_pos"] - raw["robot0_eef_pos"])
        proprio = proprio_from_obs(raw)
        images = {}
        for condition in conditions:
            axis, m = condition.split("/")
            pair = render_pair(
                scene, sample["state"], axis, float(m), seed=PROBE_SEED + index, key=index
            )
            images["clean"] = pair["clean"]
            images[condition] = pair["perturbed"]
            wrist = pair["wrist"]
        for name, image in images.items():
            captured = patcher.capture(
                runner.prepare(image, wrist, proprio, scene.language), vision_blocks
            )
            features.setdefault(f"{name}/pixels", []).append(pool_pixels(image))
            for site in sites:
                features.setdefault(f"{name}/{site}", []).append(pool(captured[site], GRID))
    scene.env.close()
    arrays = {f"label/{k}": np.stack(v).astype(np.float32) for k, v in labels.items()}
    arrays.update({f"feature/{k}": np.stack(v) for k, v in features.items()})
    arrays["sites"] = np.array(["pixels", *sites])
    temporary = output.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.rename(output)
    return f"wrote {output.name}"


def r_squared(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Mean over output dimensions of 1 − SSE/SST."""
    sse = ((truth - prediction) ** 2).sum(axis=0)
    sst = ((truth - truth.mean(axis=0)) ** 2).sum(axis=0)
    return float(np.mean(1.0 - sse / sst))


def held_out_predictions(
    train_x: np.ndarray, test_x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Fit on train_x and predict test_x, with whole tasks held out in each fold."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    prediction = np.zeros_like(y)
    for train, test in GroupKFold(FOLDS).split(train_x, y, groups):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        model.fit(train_x[train].astype(np.float32), y[train])
        prediction[test] = model.predict(test_x[test].astype(np.float32))
    return prediction


def task_bootstrap(truth, prediction, groups, resamples: int, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    tasks = np.unique(groups)
    index = {task: np.flatnonzero(groups == task) for task in tasks}
    values = []
    for _ in range(resamples):
        rows = np.concatenate([index[t] for t in rng.choice(tasks, size=len(tasks))])
        values.append(r_squared(truth[rows], prediction[rows]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def task_means(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Each row's task mean, same shape as values."""
    values = values.astype(np.float32)
    means = {task: values[groups == task].mean(axis=0) for task in np.unique(groups)}
    return np.stack([means[task] for task in groups])


def analyze(
    files: list[Path], conditions: list[str], resamples: int, centering: str = "none"
) -> pd.DataFrame:
    """Probe R² per site. centering="task" subtracts each task's mean from labels and features.

    With task centering the probe only has to track variation within a task, not predict a held-out
    task's layout. Perturbed features are centered by the clean task means for transfer (so the
    probe gets no help from the perturbed statistics) and by their own task means for refit.
    """
    rows = []
    for suite in sorted({f.stem.rsplit("_task", 1)[0] for f in files}):
        loaded = [np.load(f) for f in files if f.stem.startswith(suite)]
        sites = [str(s) for s in loaded[0]["sites"]]
        groups = np.concatenate(
            [np.full(len(d["label/target_pos"]), i) for i, d in enumerate(loaded)]
        )

        def stack(key: str, data=loaded) -> np.ndarray:
            return np.concatenate([d[key] for d in data])

        for target in TARGETS:
            y = stack(f"label/{target}")
            if centering == "task":
                y = y - task_means(y, groups)
            for depth, site in enumerate(sites):
                clean = stack(f"feature/clean/{site}")
                clean_means = task_means(clean, groups) if centering == "task" else 0.0
                clean = clean - clean_means
                base = held_out_predictions(clean, clean, y, groups)
                low, high = task_bootstrap(y, base, groups, resamples)
                common = {
                    "suite": suite,
                    "target": target,
                    "site": site,
                    "depth": depth,
                    "centering": centering,
                }
                rows.append(
                    {
                        **common,
                        "axis": "clean",
                        "mode": "clean",
                        "r2": r_squared(y, base),
                        "low": low,
                        "high": high,
                    }
                )
                for condition in conditions:
                    axis, m = condition.split("/")
                    raw = stack(f"feature/{condition}/{site}")
                    own_means = task_means(raw, groups) if centering == "task" else 0.0
                    for mode in ("transfer", "refit"):
                        perturbed = raw - (clean_means if mode == "transfer" else own_means)
                        train = clean if mode == "transfer" else perturbed
                        prediction = held_out_predictions(train, perturbed, y, groups)
                        low, high = task_bootstrap(y, prediction, groups, resamples)
                        rows.append(
                            {
                                **common,
                                "axis": axis,
                                "m": float(m),
                                "mode": mode,
                                "r2": r_squared(y, prediction),
                                "low": low,
                                "high": high,
                            }
                        )
        print(f"analyzed {suite}", flush=True)
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, path: Path, target: str, ylabel: str = "Held-out-task R²") -> None:
    import matplotlib.pyplot as plt

    from faultline_vla.viz import (
        AXIS_LABELS,
        SUITE_COLORS,
        SUITE_LABELS,
        TEXT_SECONDARY,
        apply_style,
    )

    apply_style()
    table = table[(table.target == target) & (table.site != "pixels")]
    axes_names = [a for a in table.axis.unique() if a != "clean"]
    figure, grid = plt.subplots(
        2, len(axes_names), figsize=(3.3 * len(axes_names), 6.2), sharey=True, squeeze=False
    )
    for row, suite in enumerate(SUITE_COLORS):
        color = SUITE_COLORS[suite]
        clean = table[(table.suite == suite) & (table["mode"] == "clean")].sort_values("depth")
        for column, axis in enumerate(axes_names):
            panel = grid[row, column]
            panel.plot(clean.depth, clean.r2, color=TEXT_SECONDARY, linewidth=1.2, label="Clean")
            for mode, style in (("transfer", "-"), ("refit", "--")):
                line = table[
                    (table.suite == suite) & (table.axis == axis) & (table["mode"] == mode)
                ].sort_values("depth")
                panel.plot(
                    line.depth, line.r2, color=color, linestyle=style, label=mode.capitalize()
                )
            labels = [
                s.replace("vision_", "v").replace("vlm_", "L").replace("connector", "C")
                for s in clean.site
            ]
            panel.set_xticks(clean.depth.tolist()[::2], labels[::2])
            panel.set_title(f"{SUITE_LABELS[suite]}: {AXIS_LABELS[axis]}", loc="left", fontsize=9.5)
            # Transfer R² falls far below 0 at some depths; values under the floor are clipped.
            panel.set_ylim(-1.0, 1.0)
            panel.axhline(0, color=TEXT_SECONDARY, linewidth=0.6)
        grid[row, 0].set_ylabel(ylabel)
        grid[row, 0].legend(fontsize=8)
    figure.supxlabel(
        "Depth: vision blocks (v), connector (C), VLM layers (L). R² below −1 is clipped.",
        color=TEXT_SECONDARY,
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--conditions", nargs="+", required=True, help="Written as axis/m")
    parser.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object"])
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--states-per-task", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--task-limit", type=float, default=7200)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--data-dir", type=Path, default=Path("data/stage3_probes"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    config = {
        "conditions": args.conditions,
        "states_per_task": args.states_per_task,
        "tasks": args.tasks,
        "grid": GRID,
        "seed": PROBE_SEED,
    }
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output_dir = args.data_dir / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    jobs = [
        (
            suite,
            task,
            args.conditions,
            args.states_per_task,
            output_dir / f"{suite}_task{task}.npz",
            args.threads,
            args.device,
        )
        for suite in args.suites
        for task in range(args.tasks)
    ]
    hours = len(jobs) * args.states_per_task * SECONDS_PER_STATE / args.workers / 3600
    print(f"probes {tag}: {len(jobs)} tasks, estimated {hours:.1f} h", flush=True)
    start = time.perf_counter()
    failed = run_jobs(
        collect_task, jobs, lambda job: job[4], args.workers, args.task_limit, args.retries
    )
    if failed:
        raise SystemExit(f"{len(failed)} tasks never finished; rerun to resume")
    print(f"features took {(time.perf_counter() - start) / 3600:.2f} h", flush=True)
    # "none" is the planned analysis. "task" was added after it, because held-out tasks have unseen
    # layouts and clean R² was too low to compare against (see reports/stage3.md).
    files = [job[4] for job in jobs]
    table = pd.concat(
        [
            analyze(files, args.conditions, args.resamples, centering)
            for centering in ("none", "task")
        ]
    )
    table["config_hash"] = tag
    table.to_csv(args.results_dir / "tables" / f"stage3_probes_{tag}.csv", index=False)
    for target in TARGETS:
        plot(
            table[table.centering == "none"],
            args.results_dir / "figs" / f"stage3_probes_{target}.png",
            target,
        )
        plot(
            table[table.centering == "task"],
            args.results_dir / "figs" / f"stage3_probes_{target}_within.png",
            target,
            "Within-task R², held-out tasks",
        )
    summary = table[table.site.isin(["pixels", "connector", table.site.iloc[-1]])]
    index = ["centering", "suite", "target", "axis", "mode"]
    print(summary.pivot_table(index=index, columns="site", values="r2").round(2).to_string())


if __name__ == "__main__":
    main()
