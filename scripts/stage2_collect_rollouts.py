"""Run clean closed-loop episodes with our own input pipeline and save simulator states.

The success rate also validates the pipeline against LeRobot's evaluator.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

SUITES = ("libero_spatial", "libero_object")
MAX_STEPS = 280
SETTLE_STEPS = 10
NOOP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def run_episode(runner, scene, init_index: int, seed: int, action_steps: int) -> dict:
    from faultline_vla.envs.paired import IMAGE_SIZE
    from faultline_vla.models.loaders import proprio_from_obs

    env = scene.env
    env.seed(seed)
    env.reset()
    raw = env.set_init_state(scene.init_states[init_index])
    for _ in range(SETTLE_STEPS):
        raw, _, _, _ = env.step(NOOP)
    for robot in env.robots:
        robot.controller.use_delta = True

    states, queue = [], []
    success = False
    calls = 0
    for step in range(MAX_STEPS):
        states.append(env.env.sim.get_state().flatten())
        if not queue:
            agentview = raw["agentview_image"][::-1, ::-1].copy()
            wrist = raw["robot0_eye_in_hand_image"][::-1, ::-1].copy()
            assert agentview.shape[0] == IMAGE_SIZE
            chunk = runner.predict(
                agentview,
                wrist,
                proprio_from_obs(raw),
                scene.language,
                runner.make_noise(seed + step),
            )
            queue = list(chunk[:action_steps])
            calls += 1
        raw, _, done, _ = env.step(queue.pop(0))
        if env.check_success():
            success = True
            break
        if done:
            break
    return {"states": np.stack(states), "success": success, "steps": step + 1, "calls": calls}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("data/stage2/rollouts"))
    args = parser.parse_args()

    # Load torch before creating the OSMesa context, the same order LeRobot's evaluator uses.
    from faultline_vla.models.loaders import PolicyRunner

    runner = PolicyRunner(args.device, args.threads)

    from faultline_vla.envs.paired import make_scene

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suite in SUITES:
        for task_id in range(args.tasks):
            output = args.output_dir / f"{suite}_task{task_id}.npz"
            if output.exists():
                print(json.dumps({"skip": output.name}), flush=True)
                continue
            scene = make_scene(suite, task_id)
            arrays, outcomes = {}, []
            for episode in range(args.episodes):
                start = time.perf_counter()
                result = run_episode(runner, scene, episode, args.seed + episode, args.action_steps)
                arrays[f"states_{episode}"] = result["states"]
                outcomes.append(result["success"])
                print(
                    json.dumps(
                        {
                            "suite": suite,
                            "task": task_id,
                            "episode": episode,
                            "success": result["success"],
                            "steps": result["steps"],
                            "calls": result["calls"],
                            "seconds": round(time.perf_counter() - start, 1),
                        }
                    ),
                    flush=True,
                )
            scene.env.close()
            arrays["success"] = np.array(outcomes)
            temporary = output.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.rename(output)


if __name__ == "__main__":
    main()
