"""Run LeRobot's evaluator with the policy converted to float32.

The checkpoint loads as bfloat16, which is very slow on this CPU, and oneDNN's matmul
path is about 2.5x slower than BLAS here. Everything else is LeRobot's own evaluation
path. Unrecognized arguments go to `lerobot-eval`.
"""

from __future__ import annotations

import argparse
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=6)
    args, passthrough = parser.parse_known_args()

    torch.set_num_threads(args.threads)
    if args.device == "cpu":
        torch.backends.mkldnn.enabled = False

    from lerobot.scripts import lerobot_eval

    make_policy = lerobot_eval.make_policy

    def make_float32_policy(*policy_args, **policy_kwargs):
        policy = make_policy(*policy_args, **policy_kwargs)
        policy.float()
        return policy

    lerobot_eval.make_policy = make_float32_policy
    sys.argv = [sys.argv[0], f"--policy.device={args.device}", *passthrough]
    lerobot_eval.main()


if __name__ == "__main__":
    main()
