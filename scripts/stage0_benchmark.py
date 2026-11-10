"""Stage 0 timing: rendering, simulation, and policy throughput across worker layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path

import torch

CHECKPOINT = "HuggingFaceVLA/smolvla_libero"
STATE_DIM = 32
IMAGE_SIZE = 512
LANGUAGE_TOKENS = 20


def load_model(device: str):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if device == "cpu":
        # oneDNN's matmul path is about 2.5x slower than BLAS on this CPU.
        torch.backends.mkldnn.enabled = False
    policy = SmolVLAPolicy.from_pretrained(CHECKPOINT)
    policy.to(device)
    policy.eval()
    policy.float()
    return policy


def make_inputs(device: str, seed: int):
    generator = torch.Generator().manual_seed(seed)
    images = [
        (torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator) * 2 - 1).to(device)
        for _ in range(2)
    ]
    image_masks = [torch.ones(1, dtype=torch.bool, device=device)] * 2
    tokens = torch.randint(0, 1000, (1, LANGUAGE_TOKENS), generator=generator).to(device)
    token_masks = torch.ones(1, LANGUAGE_TOKENS, dtype=torch.bool, device=device)
    state = torch.zeros(1, STATE_DIM, device=device)
    return images, image_masks, tokens, token_masks, state


def worker(device: str, threads: int, passes: int, seed: int, barrier, queue) -> None:
    torch.set_num_threads(threads)
    policy = load_model(device)
    inputs = make_inputs(device, seed)
    with torch.inference_mode():
        policy.model.sample_actions(*inputs)
        barrier.wait()
        start = time.perf_counter()
        for _ in range(passes):
            policy.model.sample_actions(*inputs)
    queue.put(time.perf_counter() - start)


def measure_layout(device: str, workers: int, threads: int, passes: int) -> dict:
    context = mp.get_context("spawn")
    barrier = context.Barrier(workers)
    queue = context.Queue()
    processes = [
        context.Process(target=worker, args=(device, threads, passes, i, barrier, queue))
        for i in range(workers)
    ]
    for process in processes:
        process.start()
    elapsed = [queue.get() for _ in processes]
    for process in processes:
        process.join()
    total_passes = workers * passes
    result = {
        "workers": workers,
        "threads": threads,
        "passes_per_worker": passes,
        "slowest_worker_s": max(elapsed),
        "seconds_per_pass_aggregate": max(elapsed) / total_passes,
        "seconds_per_pass_worker": statistics.mean(elapsed) / passes,
    }
    print(json.dumps(result), flush=True)
    return result


def measure_components(device: str, threads: int, passes: int) -> dict:
    torch.set_num_threads(threads)
    policy = load_model(device)
    model = policy.model
    backbone = model.vlm_with_expert
    timings: dict[str, list[float]] = {"embed_image": [], "backbone": []}

    def timed(function, key):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            output = function(*args, **kwargs)
            timings[key].append(time.perf_counter() - start)
            return output

        return wrapper

    backbone.embed_image = timed(backbone.embed_image, "embed_image")
    backbone.forward = timed(backbone.forward, "backbone")
    inputs = make_inputs(device, seed=0)
    totals = []
    with torch.inference_mode():
        model.sample_actions(*inputs)
        for _ in range(passes):
            timings["embed_image"].clear()
            timings["backbone"].clear()
            start = time.perf_counter()
            model.sample_actions(*inputs)
            totals.append(time.perf_counter() - start)
    result = {
        "threads": threads,
        "total_s": statistics.mean(totals),
        "embed_image_each_s": statistics.mean(timings["embed_image"]),
        "prefill_s": timings["backbone"][0],
        "denoise_step_s": statistics.mean(timings["backbone"][1:]),
        "num_vlm_layers": backbone.num_vlm_layers,
        "num_expert_layers": backbone.num_expert_layers,
        "parameters_millions": sum(p.numel() for p in policy.parameters()) / 1e6,
    }
    print(json.dumps(result), flush=True)
    return result


def measure_simulation(steps: int, size: int) -> dict:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    start = time.perf_counter()
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=size, camera_widths=size)
    env.seed(0)
    env.reset()
    env.set_init_state(suite.get_task_init_states(0)[0])
    setup = time.perf_counter() - start

    action = [0.0] * 7
    step_times, render_times = [], []
    for _ in range(steps):
        start = time.perf_counter()
        env.step(action)
        step_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        env.env.sim.render(camera_name="agentview", height=size, width=size)
        render_times.append(time.perf_counter() - start)
    env.close()
    result = {
        "image_size": size,
        "env_setup_s": setup,
        "env_step_with_two_camera_obs_s": statistics.median(step_times),
        "single_camera_render_s": statistics.median(render_times),
    }
    print(json.dumps(result), flush=True)
    return result


def parse_layout(text: str) -> tuple[int, int]:
    workers, threads = text.split("x")
    return int(workers), int(threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--passes", type=int, default=6)
    parser.add_argument("--sim-steps", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--layouts",
        nargs="+",
        type=parse_layout,
        default=[(1, 6), (1, 8), (2, 3), (2, 4), (2, 5), (3, 3), (4, 2)],
        help="Worker layouts written as WORKERSxTHREADS.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage0"))
    args = parser.parse_args()

    config = {k: str(v) for k, v in sorted(vars(args).items()) if k != "output_dir"}
    config_hash = hashlib.sha256(json.dumps(config).encode()).hexdigest()[:10]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # OSMesa and torch's thread pool can segfault when they share a process here.
    with mp.get_context("spawn").Pool(1) as pool:
        simulation = pool.apply(measure_simulation, (args.sim_steps, args.image_size))

    results = {
        "config": config,
        "config_hash": config_hash,
        "torch": torch.__version__,
        "simulation": simulation,
        "components": [measure_components(args.device, n, args.passes) for n in (6, 8)],
        "layouts": [
            measure_layout(args.device, workers, threads, args.passes)
            for workers, threads in args.layouts
        ],
    }
    path = args.output_dir / f"benchmark_{config_hash}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
