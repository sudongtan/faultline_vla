"""Stage 3 sanity check: restoring the clean agentview connector output should recover all action error.

The image enters the model only through the connector, so this tests the patching wiring rather than
any finding. Uses the same states, conditions and noise seed 0 as the readout-patching run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

NOISE_SEED = 0


def check(dataset: Path, conditions: list[str], states: int, threads: int, device: str) -> pd.DataFrame:
    from faultline_vla.models.loaders import PolicyRunner
    from faultline_vla.models.patching import ReadoutPatcher, action_error

    runner = PolicyRunner(device, threads)
    patcher = ReadoutPatcher(runner)
    noise = runner.make_noise(NOISE_SEED)
    rows = []
    for shard in sorted(dataset.glob("*_task*.npz")):
        suite, task = shard.stem.rsplit("_task", 1)
        with np.load(shard) as data:
            language = str(data["language"])
            for index in range(min(states, len(data["clean"]))):
                wrist, proprio = data["wrist"][index], data["proprio"][index]
                clean_batch = runner.prepare(data["clean"][index], wrist, proprio, language)
                clean_action = patcher.decode(patcher.prefill(clean_batch), noise)
                connector = patcher.capture(clean_batch, [])["connector"]
                for condition in conditions:
                    axis, m = condition.split("/")
                    batch = runner.prepare(data[f"perturbed/{condition}"][index], wrist, proprio, language)
                    baseline = action_error(patcher.decode(patcher.prefill(batch), noise), clean_action)
                    patched = patcher.decode(patcher.prefill_with_connector(batch, connector), noise)
                    rows.append(
                        {
                            "suite": suite,
                            "task_id": int(task),
                            "state_index": index,
                            "axis": axis,
                            "m": float(m),
                            "baseline_error": baseline,
                            "patched_error": action_error(patched, clean_action),
                        }
                    )
        print(f"done {shard.stem}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", required=True, help="Written as axis/m")
    parser.add_argument("--states-per-task", type=int, default=4)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    shards = len(list(args.dataset.glob("*_task*.npz")))
    cases = shards * args.states_per_task * len(args.conditions)
    print(f"{cases} cases, estimated {cases * 5 / 3600:.1f} h", flush=True)
    start = time.perf_counter()
    table = check(args.dataset, args.conditions, args.states_per_task, args.threads, args.device)
    table["recovery"] = 1.0 - table["patched_error"] / table["baseline_error"]
    path = args.results_dir / "tables" / f"stage3_connector_check_{args.dataset.name}.csv"
    table.to_csv(path, index=False)
    summary = table.groupby(["suite", "axis", "m"]).agg(
        recovery=("recovery", "mean"),
        worst=("recovery", "min"),
        max_patched_error=("patched_error", "max"),
    )
    print(summary.to_string(), flush=True)
    print(f"took {(time.perf_counter() - start) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
