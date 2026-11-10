# faultline_vla

When a Vision-Language-Action model fails under a visual change such as lighting, texture, camera or clutter, where inside the network does the failure happen: the vision encoder, the language backbone or the action head?

This project investigates that question using [SmolVLA](https://arxiv.org/abs/2506.01844) on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), then uses the diagnosis to build a cheap, targeted fix.

**Status:** early work in progress.

| Stage | Report |
|---|---|
| CPU setup and reproduction | [reports/stage0.md](reports/stage0.md) |
| Perturbation toolkit | [reports/stage1.md](reports/stage1.md) |
| Open-loop sensitivity and closed-loop screen | [reports/stage2.md](reports/stage2.md) |
