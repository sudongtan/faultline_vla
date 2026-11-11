"""Stage 3: readout patching and representation drift. Resumable per task."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from faultline_vla.jobs import run_jobs

VISION_BLOCKS = (2, 5, 8, 11)
BIN_SIZE = 4
SECONDS_PER_CASE = 55  # Measured for one condition and noise seed at 3 threads.
MAX_WINDOW_BINS = 4
RECOVERY_TARGET = 0.8
BACKBONE_GAIN_THRESHOLD = 1.25


def linear_cka(first, second) -> float:
    """Linear CKA between two (tokens, dim) matrices (Kornblith et al. 2019)."""
    import torch

    first = first - first.mean(dim=0, keepdim=True)
    second = second - second.mean(dim=0, keepdim=True)
    cross = torch.linalg.matrix_norm(second.T @ first) ** 2
    denominator = torch.linalg.matrix_norm(first.T @ first) * torch.linalg.matrix_norm(
        second.T @ second
    )
    return float(cross / denominator.clamp_min(1e-12))


def cosine_distance(first, second) -> float:
    import torch

    return float(1.0 - torch.nn.functional.cosine_similarity(first, second, dim=-1).mean())


def patch_plan(bins: list[list[int]], agentview) -> list[tuple]:
    """Every patch to run, as (kind, start, end, layers, tokens)."""
    plan = []
    for start, chunk in enumerate(bins):
        for length in range(1, MAX_WINDOW_BINS + 1):
            end = start + length
            if end <= len(bins):
                plan.append(
                    (
                        "restore",
                        start,
                        end,
                        [layer for part in bins[start:end] for layer in part],
                        None,
                    )
                )
        plan.append(("induce", start, start + 1, chunk, None))
        plan.append(("restore_agentview", start, start + 1, chunk, agentview))
    return plan


def localize_shard(
    shard: Path,
    conditions: list[str],
    limit: int,
    output: Path,
    threads: int,
    device: str,
    noise_seeds: list[int],
) -> str:
    from faultline_vla.models.loaders import PolicyRunner
    from faultline_vla.models.patching import (
        AGENTVIEW_TOKENS,
        ReadoutPatcher,
        action_error,
        layer_bins,
    )

    if output.exists():
        return f"skip {output.name}"
    runner = PolicyRunner(device, threads)
    patcher = ReadoutPatcher(runner)
    plan = patch_plan(layer_bins(patcher.num_layers, BIN_SIZE), AGENTVIEW_TOKENS)
    noises = {seed: runner.make_noise(seed) for seed in noise_seeds}
    suite, task = shard.stem.rsplit("_task", 1)
    patch_rows, drift_rows = [], []
    with np.load(shard) as data:
        language = str(data["language"])
        count = min(limit, len(data["clean"]))
        for index in range(count):
            wrist, proprio = data["wrist"][index], data["proprio"][index]
            clean_batch = runner.prepare(data["clean"][index], wrist, proprio, language)
            clean = patcher.prefill(clean_batch)
            clean_actions = {seed: patcher.decode(clean, noise) for seed, noise in noises.items()}
            clean_features = patcher.capture(clean_batch, VISION_BLOCKS)
            other = (index + 1) % len(data["clean"])
            other_batch = runner.prepare(
                data["clean"][other], data["wrist"][other], data["proprio"][other], language
            )
            other_features = patcher.capture(other_batch, VISION_BLOCKS)
            info = {
                "suite": suite,
                "task_id": int(task),
                "state_index": index,
                "source": str(data["source"][index]),
                "phase": str(data["phase"][index]),
            }
            for condition in conditions:
                axis, m = condition.split("/")
                batch = runner.prepare(
                    data[f"perturbed/{condition}"][index], wrist, proprio, language
                )
                perturbed = patcher.prefill(batch)
                hybrids = [
                    patcher.hybrid(clean, perturbed, layers)
                    if kind == "induce"
                    else patcher.hybrid(perturbed, clean, layers, tokens)
                    for kind, _, _, layers, tokens in plan
                ]
                for seed, noise in noises.items():
                    clean_action = clean_actions[seed]
                    baseline = action_error(patcher.decode(perturbed, noise), clean_action)
                    for (kind, start, end, _, _), hybrid in zip(plan, hybrids, strict=True):
                        patch_rows.append(
                            {
                                **info,
                                "axis": axis,
                                "m": float(m),
                                "noise_seed": seed,
                                "baseline_error": baseline,
                                "kind": kind,
                                "start": start,
                                "end": end,
                                "error": action_error(patcher.decode(hybrid, noise), clean_action),
                            }
                        )
                features = patcher.capture(batch, VISION_BLOCKS)
                for depth, name in enumerate(features):
                    drift_rows.append(
                        {
                            **info,
                            "axis": axis,
                            "m": float(m),
                            "layer": name,
                            "depth": depth,
                            "cosine": cosine_distance(clean_features[name], features[name]),
                            "cka": linear_cka(clean_features[name], features[name]),
                            "state_cosine": cosine_distance(
                                clean_features[name], other_features[name]
                            ),
                        }
                    )
    temporary = output.with_suffix(".tmp.parquet")
    pd.DataFrame(patch_rows).to_parquet(temporary)
    pd.DataFrame(drift_rows).to_parquet(output.with_suffix(".drift.parquet"))
    temporary.rename(output)
    return f"wrote {output.name}"


def recovery(group: pd.DataFrame) -> float:
    return 1.0 - group["error"].sum() / group["baseline_error"].sum()


def bootstrap_recovery(
    group: pd.DataFrame, resamples: int, seed: int
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    by_task = group.groupby("task_id")[["error", "baseline_error"]].sum().to_numpy()
    picks = rng.integers(0, len(by_task), size=(resamples, len(by_task)))
    samples = 1.0 - by_task[picks, 0].sum(axis=1) / by_task[picks, 1].sum(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return 1.0 - by_task[:, 0].sum() / by_task[:, 1].sum(), low, high


def analyze(
    patches: pd.DataFrame, drift: pd.DataFrame, results: Path, tag: str, resamples: int
) -> dict:
    tables = results / "tables"
    rows = []
    for keys, group in patches.groupby(["suite", "axis", "m", "kind", "start", "end"]):
        value, low, high = bootstrap_recovery(group, resamples, 0)
        rows.append(
            dict(zip(["suite", "axis", "m", "kind", "start", "end"], keys, strict=True))
            | {"recovery": value, "low": low, "high": high}
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(tables / f"stage3_patching_{tag}.csv", index=False)

    drift = drift.copy()
    drift["normalized"] = drift["cosine"] / drift["state_cosine"].clip(lower=1e-6)
    profile = drift.groupby(["suite", "axis", "m", "layer", "depth"], as_index=False)[
        ["cosine", "cka", "normalized"]
    ].mean()
    profile.to_csv(tables / f"stage3_drift_{tag}.csv", index=False)

    calls = []
    for (suite, axis, m), group in summary[summary.kind == "restore"].groupby(
        ["suite", "axis", "m"]
    ):
        group = group.assign(length=group.end - group.start)
        passing = group[group.recovery >= RECOVERY_TARGET].sort_values(
            ["length", "recovery"], ascending=[True, False]
        )
        line = profile[
            (profile.suite == suite) & (profile.axis == axis) & (profile.m == m)
        ].sort_values("depth")
        connector = line[line.layer == "connector"]["normalized"].iloc[0]
        final = line[line.layer.str.startswith("vlm_")]["normalized"].iloc[-1]
        gain = final / max(connector, 1e-6)
        if passing.empty:
            window = None
            verdict = "diffuse"
        else:
            best = passing.iloc[0]
            window = [int(best.start), int(best.end)]
            verdict = "backbone" if gain > BACKBONE_GAIN_THRESHOLD else "encoder"
        calls.append(
            {
                "suite": suite,
                "axis": axis,
                "m": m,
                "window_bins": window,
                "window_recovery": None if passing.empty else float(passing.iloc[0].recovery),
                "best_single_bin": float(group[group.length == 1].recovery.max()),
                "connector_drift": float(connector),
                "final_drift": float(final),
                "drift_gain": float(gain),
                "verdict": verdict,
            }
        )
    calls_table = pd.DataFrame(calls)
    calls_table.to_csv(tables / f"stage3_calls_{tag}.csv", index=False)
    print(calls_table.to_string(), flush=True)
    return {"summary": summary, "profile": profile, "calls": calls_table}


def plot(outputs: dict, figures: Path, bins: int, bin_size: int, suffix: str = "") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from faultline_vla.viz import AXIS_LABELS, SUITE_COLORS, SUITE_LABELS, apply_style

    apply_style()
    colormap = LinearSegmentedColormap.from_list(
        "recovery", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]
    )
    summary = outputs["summary"]
    single = summary[(summary.kind == "restore") & (summary.end - summary.start == 1)]
    keys = sorted({(row.axis, row.m) for row in single.itertuples()})
    figure, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(keys) + 1.8), sharey=True)
    for plot_axis, suite in zip(axes, SUITE_COLORS, strict=True):
        matrix = np.full((len(keys), bins), np.nan)
        for row in single[single.suite == suite].itertuples():
            matrix[keys.index((row.axis, row.m)), int(row.start)] = row.recovery
        image = plot_axis.imshow(
            np.clip(matrix, 0, 1), aspect="auto", cmap=colormap, vmin=0, vmax=1
        )
        plot_axis.set_xticks(range(bins), [f"{i * bin_size}" for i in range(bins)])
        plot_axis.set_yticks(range(len(keys)), [f"{AXIS_LABELS[a]}, m={m}" for a, m in keys])
        plot_axis.set_title(SUITE_LABELS[suite], loc="left")
        plot_axis.set_xlabel("First VLM layer in the restored 4-layer readout")
        plot_axis.grid(False)
    figure.colorbar(image, ax=axes, label="Share of action error recovered", shrink=0.85)
    figure.savefig(figures / f"stage3_break_heatmap{suffix}.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    profile = outputs["profile"]
    pairs = sorted(
        {
            (row.axis, row.m)
            for row in profile.itertuples()
            if row.m == profile[profile.axis == row.axis].m.max()
        }
    )
    figure, axes = plt.subplots(
        1, len(pairs), figsize=(3.4 * len(pairs), 3.6), sharey=True, squeeze=False
    )
    for plot_axis, (axis, m) in zip(axes[0], pairs, strict=True):
        for suite, color in SUITE_COLORS.items():
            line = profile[
                (profile.suite == suite) & (profile.axis == axis) & (profile.m == m)
            ].sort_values("depth")
            plot_axis.plot(line.depth, line.normalized, color=color, label=SUITE_LABELS[suite])
        plot_axis.set_title(f"{AXIS_LABELS[axis]}, m={m}", loc="left")
        plot_axis.set_xlabel("Depth: vision blocks, connector, VLM layers")
    axes[0, 0].set_ylabel("Drift relative to between-state distance")
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(figures / f"stage3_drift{suffix}.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        required=True,
        help="Written as axis/m, for example camera_pose/1.0",
    )
    parser.add_argument(
        "--states-per-task",
        type=int,
        default=4,
        help="The default keeps both demo and rollout states",
    )
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument(
        "--shard-limit",
        type=float,
        default=7200,
        help="Seconds before a shard is killed and retried",
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--figure-suffix",
        default="",
        help="Appended to figure names so runs don't overwrite each other",
    )
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip patching and analyze the shards already saved",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    config = {
        "conditions": args.conditions,
        "states_per_task": args.states_per_task,
        "bin_size": BIN_SIZE,
        "noise_seeds": args.noise_seeds,
    }
    import hashlib

    tag = (
        args.dataset.name
        + "_"
        + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    )
    output_dir = args.results_dir / "stage3" / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    shards = sorted(args.dataset.glob("*_task*.npz"))
    cases = len(shards) * args.states_per_task * len(args.conditions) * len(args.noise_seeds)
    hours = cases * SECONDS_PER_CASE / args.workers / 3600
    print(f"{cases} patch cases over {args.workers} workers, estimated {hours:.1f} h", flush=True)

    start = time.perf_counter()
    jobs = [
        (
            shard,
            args.conditions,
            args.states_per_task,
            output_dir / f"{shard.stem}.parquet",
            args.threads,
            args.device,
            args.noise_seeds,
        )
        for shard in shards
    ]
    if args.analyze_only:
        jobs = [job for job in jobs if job[3].exists()]
        print(f"analyzing {len(jobs)} saved shards of {len(shards)}", flush=True)
    else:
        failed = run_jobs(
            localize_shard, jobs, lambda job: job[3], args.workers, args.shard_limit, args.retries
        )
        if failed:
            raise SystemExit(f"{len(failed)} shards never finished; rerun to resume")
        print(f"patching took {(time.perf_counter() - start) / 3600:.2f} h", flush=True)
    patches = pd.concat([pd.read_parquet(job[3]) for job in jobs], ignore_index=True)
    drift = pd.concat(
        [pd.read_parquet(job[3].with_suffix(".drift.parquet")) for job in jobs], ignore_index=True
    )
    tables = args.results_dir / "tables"
    patches.to_parquet(tables / f"stage3_raw_patch_{tag}.parquet")
    drift.to_parquet(tables / f"stage3_raw_drift_{tag}.parquet")
    outputs = analyze(patches, drift, args.results_dir, tag, args.resamples)
    plot(
        outputs,
        args.results_dir / "figs",
        bins=int(patches["start"].max()) + 1,
        bin_size=BIN_SIZE,
        suffix=args.figure_suffix,
    )


if __name__ == "__main__":
    main()
