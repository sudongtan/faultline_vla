"""Run long jobs in separate processes so that one killed worker cannot hang the run."""

from __future__ import annotations

import multiprocessing as mp
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path


def run_jobs(
    target: Callable,
    jobs: list[tuple],
    output_of: Callable[[tuple], Path],
    workers: int,
    limit: float,
    retries: int,
) -> list[tuple]:
    """Run ``target(*job)`` per job, with a time limit and retries. Returns jobs that never finished.

    A job counts as finished when ``output_of(job)`` exists. Unlike ``multiprocessing.Pool``, this
    survives a worker being killed, for example by the OOM killer.
    """
    context = mp.get_context("spawn")
    pending = deque((job, 0) for job in jobs)
    running: dict = {}
    failed = []
    while pending or running:
        while pending and len(running) < workers:
            job, attempt = pending.popleft()
            process = context.Process(target=target, args=job)
            process.start()
            running[process] = (job, attempt, time.monotonic())
        for process, (job, attempt, started) in list(running.items()):
            if process.is_alive() and time.monotonic() - started > limit:
                print(f"TIMEOUT {job[:5]}", flush=True)
                process.kill()
            if process.is_alive():
                continue
            process.join()
            del running[process]
            if output_of(job).exists():
                continue
            print(f"RETRY {job[:5]} exit {process.exitcode} attempt {attempt}", flush=True)
            if attempt < retries:
                pending.append((job, attempt + 1))
            else:
                print(f"FAILED {job[:5]}", flush=True)
                failed.append(job)
        time.sleep(1)
    return failed
