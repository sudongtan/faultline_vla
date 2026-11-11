"""Stage 2: measure how much each perturbation changes SmolVLA's actions. Resumable per task."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

FLOOR_SEEDS = (1, 2, 3)
BASE_SEED = 0
METRICS = ("trans_first", "rot_first", "grip_first", "trans_chunk", "rot_chunk", "grip_chunk")


def compare(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Compare two action chunks with shape (steps, 7)."""
    translation = np.linalg.norm(prediction[:, :3] - reference[:, :3], axis=1)
    rotation = np.linalg.norm(prediction[:, 3:6] - reference[:, 3:6], axis=1)
    flips = np.sign(prediction[:, 6]) != np.sign(reference[:, 6])
    return {
        "trans_first": float(translation[0]),
        "rot_first": float(rotation[0]),
        "grip_first": float(flips[0]),
        "trans_chunk": float(translation.mean()),
        "rot_chunk": float(rotation.mean()),
        "grip_chunk": float(flips.mean()),
    }


def infer_shard(shard: Path, output: Path, threads: int, device: str) -> str:
    from faultline_vla.models.loaders import PolicyRunner

    if output.exists():
        return f"skip {output.name}"
    runner = PolicyRunner(device, threads)
    suite, task = shard.stem.rsplit("_task", 1)
    rows = []
    chunks: dict[str, np.ndarray] = {}
    with np.load(shard) as data:
        language = str(data["language"])
        conditions = [
            key.removeprefix("perturbed/") for key in data.files if key.startswith("perturbed/")
        ]
        count = len(data["clean"])
        base_noise = runner.make_noise(BASE_SEED)
        for index in range(count):
            wrist = data["wrist"][index]
            proprio = data["proprio"][index]
            info = {
                "suite": suite,
                "task_id": int(task),
                "state_index": index,
                "source": str(data["source"][index]),
                "phase": str(data["phase"][index]),
            }
            clean_batch = runner.prepare(data["clean"][index], wrist, proprio, language)
            clean = runner.chunk(clean_batch, base_noise)
            chunks[f"clean/{index}"] = clean.astype(np.float32)
            for seed in FLOOR_SEEDS:
                other = runner.chunk(clean_batch, runner.make_noise(seed))
                rows.append(
                    {
                        **info,
                        "kind": "noise_floor",
                        "axis": "clean",
                        "m": 0.0,
                        "noise": seed,
                        **compare(other, clean),
                    }
                )
            demo = data["demo_action"][index]
            if not np.isnan(demo).any():
                rows.append(
                    {
                        **info,
                        "kind": "clean_vs_demo",
                        "axis": "clean",
                        "m": 0.0,
                        "noise": BASE_SEED,
                        **compare(clean[:1], demo[None]),
                    }
                )
            for condition in conditions:
                axis, m = condition.split("/")
                image = data[f"perturbed/{condition}"][index]
                perturbed = runner.predict(image, wrist, proprio, language, base_noise)
                chunks[f"{condition}/{index}"] = perturbed.astype(np.float32)
                rows.append(
                    {
                        **info,
                        "kind": "perturbed",
                        "axis": axis,
                        "m": float(m),
                        "noise": BASE_SEED,
                        **compare(perturbed, clean),
                    }
                )
                if not np.isnan(demo).any():
                    rows.append(
                        {
                            **info,
                            "kind": "perturbed_vs_demo",
                            "axis": axis,
                            "m": float(m),
                            "noise": BASE_SEED,
                            **compare(perturbed[:1], demo[None]),
                        }
                    )
    table = pd.DataFrame(rows)
    temporary = output.with_suffix(".tmp.parquet")
    table.to_parquet(temporary)
    np.savez_compressed(output.with_suffix(".chunks.npz"), **chunks)
    temporary.rename(output)
    return f"wrote {output.name}"


def bootstrap(
    table: pd.DataFrame, columns: list[str], metric: str, resamples: int, seed: int
) -> pd.DataFrame:
    """Mean of ``metric`` per group, with CIs from resampling tasks."""
    rng = np.random.default_rng(seed)
    rows = []
    for keys, group in table.groupby(columns):
        keys = keys if isinstance(keys, tuple) else (keys,)
        values = group.groupby("task_id")[metric].mean().to_numpy()
        samples = rng.choice(values, size=(resamples, len(values))).mean(axis=1)
        low, high = np.percentile(samples, [2.5, 97.5])
        rows.append(
            {
                **dict(zip(columns, keys, strict=True)),
                "metric": metric,
                "mean": values.mean(),
                "low": low,
                "high": high,
                "n_tasks": len(values),
            }
        )
    return pd.DataFrame(rows)


def analyze(
    table: pd.DataFrame, image_summary: Path, results: Path, tag: str, resamples: int
) -> None:
    import matplotlib.pyplot as plt

    from faultline_vla.envs.perturb import AXES
    from faultline_vla.viz import AXIS_LABELS, SUITE_COLORS, SUITE_LABELS, apply_style

    tables = results / "tables"
    figures = results / "figs"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    perturbed = table[table.kind == "perturbed"]
    floor = table[table.kind == "noise_floor"]
    dose = pd.concat(
        [bootstrap(perturbed, ["suite", "axis", "m"], metric, resamples, 0) for metric in METRICS]
    )
    floor_summary = pd.concat(
        [bootstrap(floor, ["suite"], metric, resamples, 0) for metric in METRICS]
    )
    phase = pd.concat(
        [
            bootstrap(perturbed, ["suite", "axis", "m", "phase"], metric, resamples, 0)
            for metric in ("trans_first", "rot_first")
        ]
    )
    source = pd.concat(
        [
            bootstrap(perturbed, ["suite", "axis", "m", "source"], metric, resamples, 0)
            for metric in ("trans_first", "rot_first")
        ]
    )
    demo = table[table.kind.isin(["perturbed_vs_demo", "clean_vs_demo"])]
    demo_summary = bootstrap(demo, ["suite", "kind", "axis", "m"], "trans_first", resamples, 0)
    dose.to_csv(tables / f"stage2_dose_{tag}.csv", index=False)
    floor_summary.to_csv(tables / f"stage2_floor_{tag}.csv", index=False)
    phase.to_csv(tables / f"stage2_phase_{tag}.csv", index=False)
    source.to_csv(tables / f"stage2_source_{tag}.csv", index=False)
    demo_summary.to_csv(tables / f"stage2_demo_{tag}.csv", index=False)

    verdicts = []
    for (suite, axis), group in dose[dose.metric == "trans_first"].groupby(["suite", "axis"]):
        top = group.sort_values("m").iloc[-1]
        ceiling = floor_summary[
            (floor_summary.suite == suite) & (floor_summary.metric == "trans_first")
        ]["high"].iloc[0]
        verdicts.append(
            {
                "suite": suite,
                "axis": axis,
                "error_at_max": top["mean"],
                "low": top["low"],
                "floor_high": ceiling,
                "above_floor": bool(top["low"] > ceiling),
            }
        )
    verdicts = pd.DataFrame(verdicts)
    verdicts.to_csv(tables / f"stage2_above_floor_{tag}.csv", index=False)
    print(verdicts.to_string(), flush=True)

    apply_style()
    trans = dose[dose.metric == "trans_first"]
    floor_trans = floor_summary[floor_summary.metric == "trans_first"].set_index("suite")["mean"]
    figure, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey=True)
    for plot, axis in zip(axes.ravel(), AXES, strict=True):
        for suite, color in SUITE_COLORS.items():
            data = trans[(trans.axis == axis) & (trans.suite == suite)].sort_values("m")
            plot.fill_between(data.m, data.low, data.high, color=color, alpha=0.15, lw=0)
            plot.plot(
                data.m, data["mean"], color=color, marker="o", ms=4, label=SUITE_LABELS[suite]
            )
            plot.axhline(floor_trans[suite], color=color, ls=":", lw=1)
        plot.set_title(AXIS_LABELS[axis], loc="left")
    axes[0, 0].legend(loc="upper left")
    figure.supxlabel("Perturbation magnitude m (dotted line: noise-seed floor)")
    figure.supylabel("First-action translation change vs clean image")
    figure.tight_layout()
    figure.savefig(figures / "stage2_dose_response.png", dpi=160)
    plt.close(figure)

    if image_summary.exists():
        images = pd.read_csv(image_summary)
        merged = trans.merge(images[["suite", "axis", "m", "lpips"]], on=["suite", "axis", "m"])
        top = merged.sort_values("m").groupby(["suite", "axis"]).tail(1).copy()
        # Noise is paired, so the change itself is the effect. Subtracting the unpaired floor clips every axis to 0.
        top["sensitivity"] = top["mean"] / top["lpips"]
        top.to_csv(tables / f"stage2_sensitivity_{tag}.csv", index=False)
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for plot, (suite, color) in zip(axes, SUITE_COLORS.items(), strict=True):
            data = top[top.suite == suite].sort_values("sensitivity")
            plot.barh(
                [AXIS_LABELS[a] for a in data.axis], data.sensitivity, color=color, height=0.55
            )
            plot.set_title(SUITE_LABELS[suite], loc="left")
            plot.grid(axis="y", visible=False)
            plot.set_xlabel("First-action change per unit LPIPS, m = max")
        figure.tight_layout()
        figure.savefig(figures / "stage2_sensitivity.png", dpi=160)
        plt.close(figure)


def paired_effects(table: pd.DataFrame, results: Path, tag: str, resamples: int) -> None:
    """Post-hoc, state-paired effects: change relative to each state's own floor, and error against the demo."""
    import matplotlib.pyplot as plt

    from faultline_vla.envs.perturb import AXES
    from faultline_vla.viz import AXIS_LABELS, SUITE_COLORS, SUITE_LABELS, apply_style

    state = ["suite", "task_id", "source", "state_index"]
    floor = (
        table[table.kind == "noise_floor"].groupby(state).trans_first.mean().rename("state_floor")
    )
    clean = table[table.kind == "clean_vs_demo"].set_index(state).trans_first.rename("clean_error")
    perturbed = table[table.kind == "perturbed"].join(floor, on=state)
    perturbed["floor_ratio"] = perturbed.trans_first / perturbed.state_floor
    perturbed["over_floor"] = (perturbed.trans_first > perturbed.state_floor).astype(float)
    versus_demo = table[table.kind == "perturbed_vs_demo"].join(clean, on=state)
    versus_demo["demo_error_change"] = versus_demo.trans_first - versus_demo.clean_error

    def strongest(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame.groupby(["suite", "axis"]).m.transform("max") == frame.m]

    summary = pd.concat(
        [
            bootstrap(strongest(perturbed), ["suite", "axis"], "floor_ratio", resamples, 0),
            bootstrap(strongest(perturbed), ["suite", "axis"], "over_floor", resamples, 0),
            bootstrap(strongest(versus_demo), ["suite", "axis"], "demo_error_change", resamples, 0),
        ]
    )
    summary.to_csv(results / "tables" / f"stage2_paired_{tag}.csv", index=False)
    print(summary.round(3).to_string(), flush=True)

    apply_style()
    labels = [AXIS_LABELS[axis] for axis in AXES]
    panels = (
        ("floor_ratio", "Change / own noise floor, m = max", 1.0),
        ("demo_error_change", "Added error against demo action, m = max", 0.0),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for row, (metric, label, reference) in zip(axes, panels, strict=True):
        for plot, (suite, color) in zip(row, SUITE_COLORS.items(), strict=True):
            data = (
                summary[(summary.metric == metric) & (summary.suite == suite)]
                .set_index("axis")
                .loc[list(AXES)]
            )
            positions = np.arange(len(labels))
            errors = [data["mean"] - data["low"], data["high"] - data["mean"]]
            plot.errorbar(data["mean"], positions, xerr=errors, fmt="o", ms=6, color=color, lw=2)
            plot.axvline(reference, color="#777777", ls=":", lw=1)
            plot.set_yticks(positions, labels)
            plot.invert_yaxis() if plot is row[0] else None
            plot.grid(axis="y", visible=False)
            plot.set_title(SUITE_LABELS[suite], loc="left")
            plot.set_xlabel(label)
    figure.tight_layout()
    figure.savefig(results / "figs" / "stage2_paired_effects.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seconds-per-pass", type=float, default=1.7)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument(
        "--image-summary",
        type=Path,
        default=Path("results/tables/stage1_image_change_summary_fdf21ecd25.csv"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--analyze-only", action="store_true", help="Rebuild tables and figures from saved actions."
    )
    args = parser.parse_args()

    tag = args.dataset.name
    if args.analyze_only:
        table = pd.read_parquet(args.results_dir / "tables" / f"stage2_actions_{tag}.parquet")
        analyze(table, args.image_summary, args.results_dir, tag, args.resamples)
        paired_effects(table, args.results_dir, tag, args.resamples)
        return
    output_dir = args.results_dir / "stage2" / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(args.dataset.glob("*_task*.npz"))
    with np.load(shards[0]) as data:
        per_state = sum(key.startswith("perturbed/") for key in data.files) + 1 + len(FLOOR_SEEDS)
        states_per_task = len(data["clean"])
    total = per_state * states_per_task * len(shards)
    print(
        f"{total} forward passes, estimated {total * args.seconds_per_pass / 3600:.1f} h",
        flush=True,
    )

    start = time.perf_counter()
    jobs = [
        (shard, output_dir / f"{shard.stem}.parquet", args.threads, args.device) for shard in shards
    ]
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for message in pool.starmap(infer_shard, jobs, chunksize=1):
            print(message, flush=True)
    print(f"inference took {(time.perf_counter() - start) / 3600:.2f} h", flush=True)
    table = pd.concat([pd.read_parquet(job[1]) for job in jobs], ignore_index=True)
    table["config_hash"] = tag
    table.to_parquet(args.results_dir / "tables" / f"stage2_actions_{tag}.parquet")
    analyze(table, args.image_summary, args.results_dir, tag, args.resamples)
    paired_effects(table, args.results_dir, tag, args.resamples)


if __name__ == "__main__":
    main()
