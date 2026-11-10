# SmolVLA model notes

Facts for `HuggingFaceVLA/smolvla_libero` (snapshot `6721902`), read from the checkpoint and LeRobot 0.7.0 at load time.

## Anatomy

| Part | Value |
|---|---|
| Parameters | 604.9M |
| Stored dtype | `bfloat16` (converted to `float32` for CPU inference) |
| Base VLM | `HuggingFaceTB/SmolVLM2-500M-Instruct` |
| Vision encoder | SigLIP, 12 blocks, 512×512 input |
| VLM text layers | 32 (`num_vlm_layers: 0` means no truncation) |
| Action expert layers | 32, one per VLM layer |
| Expert width | 0.5× VLM width |
| Trained parts | Action expert, state projection and action projections (`train_expert_only: true`) |
| Frozen parts | Vision encoder, connector and VLM text layers |
| Denoising | Flow matching, 10 Euler steps |
| Chunk | 50 actions, 7-D |
| `n_action_steps` | 1 in the checkpoint |

## Inputs

| Key | Source | Shape |
|---|---|---|
| `observation.images.image` | Agentview camera, rotated 180° by LeRobot | 3×256×256, resized with padding to 512×512 |
| `observation.images.image2` | Wrist camera, rotated 180° by LeRobot | 3×256×256, resized with padding to 512×512 |
| `observation.state` | End-effector position (3), axis-angle (3), gripper joints (2) | 8, padded to 32 |
| `task` | Instruction, tokenized to at most 48 tokens | variable |

The action is a relative end-effector command: position (3), rotation (3), gripper (1).

## Token layout

The prefix is ordered as follows. Each image gives 64 tokens at 960 dimensions.

| Position | Tokens |
|---|---|
| 0–63 | Agentview (`image`) |
| 64–127 | Wrist (`image2`) |
| 128 onward | Language |
| Last | State |

Check camera order against the batch's image keys when building masks.

## How the expert reads the VLM

1. The prefix passes through the VLM once. Each layer's post-RoPE keys and values are stored in a `DynamicCache`.
2. Each denoising step reads that fixed cache. The cache is cropped back to the prefix length after each step.
3. Even layers (`layer % 2 == 0`) run joint attention: the expert's queries attend over `[cached prefix; suffix]`.
4. Odd layers run cross-attention: cached keys and values pass through the expert's `k_proj` and `v_proj`.

Layer 0's keys and values derive from the connector output plus the first layer's projections, so the readout index runs from near the vision encoder's output to the deepest VLM layer.

## Hook points

| Point | Module or object |
|---|---|
| Vision encoder blocks | `vlm_with_expert.vlm.model.vision_model.encoder.layers[i]` |
| Connector output | `vlm_with_expert.vlm.model.connector` |
| VLM layer inputs | Hidden states inside `SmolVLMWithExpertModel.forward` |
| Expert readouts | `past_key_values.layers[i].keys` and `.values` after the prefill |

## CPU settings

`torch.backends.mkldnn.enabled = False` and `float32`. On an M1 Pro in arm64 Docker this takes about 2 s per pass. See `results/stage0/`.
