"""Stage 1: render paired images from demo states and measure image change per axis."""

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


def magnitudes_for(axis: str) -> tuple[float, ...]:
    return DISTRACTOR_MAGNITUDES if axis == "distractors" else MAGNITUDES


def sample_demo_states(path: Path, count: int, seed: int) -> np.ndarray:
    """Pick ``count`` states spread across different demonstrations."""
    import h5py

    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as handle:
        demos = sorted(handle["data"], key=lambda name: int(name.split("_")[1]))
        chosen = rng.choice(len(demos), size=count, replace=False)
        states = []
        for index in chosen:
            data = handle["data"][demos[index]]["states"]
            states.append(data[rng.integers(len(data))])
    return np.stack(states)


def render_task(suite: str, task_id: int, count: int, seed: int, output: Path) -> str:
    """Render one task's clean and perturbed images into a shard. Skips finished shards."""
    from libero.libero import benchmark

    from faultline_vla.envs.paired import make_scene, render_pair
    from faultline_vla.envs.perturb import AXES

    if output.exists():
        return f"skip {output.name}"
    task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
    states = sample_demo_states(DEMO_ROOT / suite / f"{task.name}_demo.hdf5", count, seed + task_id)
    scene = make_scene(suite, task_id)
    arrays: dict[str, np.ndarray] = {"states": states}
    for index, state in enumerate(states):
        for axis in AXES:
            for m in magnitudes_for(axis):
                pair = render_pair(scene, state, axis, m, seed=seed + index, key=index)
                arrays[f"{axis}/{m}/{index}"] = pair["perturbed"]
                arrays[f"clean/{index}"] = pair["clean"]
    scene.env.close()
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.rename(output)
    return f"wrote {output.name}"


def compute_metrics(shards: list[Path], device: str):
    import lpips
    import pandas as pd
    import torch
    from skimage.metrics import structural_similarity

    from faultline_vla.envs.perturb import AXES

    network = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def to_tensor(image: np.ndarray) -> torch.Tensor:
        return (torch.from_numpy(image).permute(2, 0, 1)[None].float() / 127.5 - 1.0).to(device)

    rows = []
    for shard in shards:
        suite, task = shard.stem.rsplit("_task", 1)
        with np.load(shard) as data:
            count = len(data["states"])
            for index in range(count):
                clean = data[f"clean/{index}"]
                clean_tensor = to_tensor(clean)
                for axis in AXES:
                    for m in magnitudes_for(axis):
                        image = data[f"{axis}/{m}/{index}"]
                        with torch.inference_mode():
                            perceptual = network(clean_tensor, to_tensor(image)).item()
                        difference = (clean.astype(float) - image.astype(float)) / 255.0
                        rows.append(
                            {
                                "suite": suite,
                                "task_id": int(task),
                                "state_index": index,
                                "axis": axis,
                                "m": m,
                                "rmse": float(np.sqrt((difference**2).mean())),
                                "dssim": 1.0
                                - float(
                                    structural_similarity(
                                        clean, image, channel_axis=2, data_range=255
                                    )
                                ),
                                "lpips": perceptual,
                            }
                        )
    return pd.DataFrame(rows)


def summarize(table, resamples: int, seed: int):
    """Mean of each metric per condition, with bootstrap CIs over tasks."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for (suite, axis, m), group in table.groupby(["suite", "axis", "m"]):
        row = {"suite": suite, "axis": axis, "m": m}
        task_means = group.groupby("task_id")[["rmse", "dssim", "lpips"]].mean()
        for metric in ("rmse", "dssim", "lpips"):
            values = task_means[metric].to_numpy()
            samples = rng.choice(values, size=(resamples, len(values))).mean(axis=1)
            row[metric] = values.mean()
            row[f"{metric}_low"], row[f"{metric}_high"] = np.percentile(samples, [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows)


def monotonic_report(summary, tolerance: float = 0.002) -> list[dict]:
    report = []
    for (suite, axis), group in summary.groupby(["suite", "axis"]):
        group = group.sort_values("m")
        values = group["lpips"].to_numpy()
        worst_drop = float(np.max(values[:-1] - values[1:])) if len(values) > 1 else 0.0
        report.append(
            {
                "suite": suite,
                "axis": axis,
                "monotone": worst_drop <= tolerance,
                "worst_drop": worst_drop,
            }
        )
    return report


def plot_curves(summary, path: Path) -> None:
    import matplotlib.pyplot as plt

    from faultline_vla.envs.perturb import AXES
    from faultline_vla.viz import AXIS_LABELS, SUITE_COLORS, SUITE_LABELS, apply_style

    apply_style()
    figure, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey=True)
    for plot, axis in zip(axes.ravel(), AXES, strict=True):
        for suite, color in SUITE_COLORS.items():
            data = summary[(summary.axis == axis) & (summary.suite == suite)].sort_values("m")
            plot.fill_between(data.m, data.lpips_low, data.lpips_high, color=color, alpha=0.15, lw=0)
            plot.plot(data.m, data.lpips, color=color, marker="o", ms=4, label=SUITE_LABELS[suite])
        plot.set_title(AXIS_LABELS[axis], loc="left")
    axes[0, 0].legend(loc="upper left")
    figure.supxlabel("Perturbation magnitude m")
    figure.supylabel("LPIPS distance from the clean image")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_contact_sheet(shard: Path, path: Path, state_index: int) -> None:
    import matplotlib.pyplot as plt

    from faultline_vla.envs.perturb import AXES
    from faultline_vla.viz import AXIS_LABELS, apply_style

    apply_style()
    columns = 6
    figure, axes = plt.subplots(len(AXES), columns, figsize=(columns * 1.45, len(AXES) * 1.45))
    with np.load(shard) as data:
        clean = data[f"clean/{state_index}"]
        for row, axis in enumerate(AXES):
            magnitudes = magnitudes_for(axis)
            cells = [clean] + [data[f"{axis}/{m}/{state_index}"] for m in magnitudes]
            labels = ["m=0"] + [f"m={m}" for m in magnitudes]
            for column in range(columns):
                plot = axes[row, column]
                plot.axis("off")
                if column < len(cells):
                    plot.imshow(cells[column])
                    if row == 0 or axis == "distractors":
                        plot.set_title(labels[column], fontsize=8)
            axes[row, 0].axis("on")
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])
            axes[row, 0].grid(False)
            for spine in axes[row, 0].spines.values():
                spine.set_visible(False)
            axes[row, 0].set_ylabel(AXIS_LABELS[axis], fontsize=8, rotation=0, ha="right", va="center")
    figure.tight_layout(pad=0.3)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--states-per-task", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--data-dir", type=Path, default=Path("data/stage1"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    config = {
        "tasks": args.tasks,
        "states_per_task": args.states_per_task,
        "seed": args.seed,
        "magnitudes": MAGNITUDES,
        "distractor_magnitudes": DISTRACTOR_MAGNITUDES,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    shard_dir = args.data_dir / config_hash
    shard_dir.mkdir(parents=True, exist_ok=True)
    tables = args.results_dir / "tables"
    figures = args.results_dir / "figs"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    jobs = [
        (suite, task, args.states_per_task, args.seed, shard_dir / f"{suite}_task{task}.npz")
        for suite in SUITES
        for task in range(args.tasks)
    ]
    print(f"config {config_hash}: rendering {len(jobs)} tasks with {args.workers} workers", flush=True)
    context = mp.get_context("spawn")
    with context.Pool(args.workers, maxtasksperchild=1) as pool:
        for message in pool.starmap(render_task, jobs, chunksize=1):
            print(message, flush=True)

    shards = [job[-1] for job in jobs]
    table = compute_metrics(shards, args.device)
    table["config_hash"] = config_hash
    table.to_parquet(tables / f"stage1_image_change_{config_hash}.parquet")
    summary = summarize(table, args.resamples, args.seed)
    summary["config_hash"] = config_hash
    summary.to_csv(tables / f"stage1_image_change_summary_{config_hash}.csv", index=False)
    report = monotonic_report(summary)
    (tables / f"stage1_monotonic_{config_hash}.json").write_text(json.dumps(report, indent=2))
    for row in report:
        print(json.dumps(row), flush=True)

    plot_curves(summary, figures / "stage1_image_change.png")
    plot_contact_sheet(shard_dir / "libero_spatial_task0.npz", figures / "stage1_contact_spatial.png", 0)
    plot_contact_sheet(shard_dir / "libero_object_task0.npz", figures / "stage1_contact_object.png", 0)
    print(f"done {config_hash}", flush=True)


if __name__ == "__main__":
    main()
