"""Tests that readout patching reproduces the model's own outputs."""

from __future__ import annotations

import numpy as np
import pytest

from faultline_vla.models.loaders import PolicyRunner
from faultline_vla.models.patching import AGENTVIEW_TOKENS, IMAGE_TOKENS, ReadoutPatcher


@pytest.fixture(scope="module")
def setup():
    runner = PolicyRunner("cpu", 8)
    rng = np.random.default_rng(0)
    clean = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    perturbed = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    proprio = rng.normal(size=8).astype(np.float32)
    language = "pick up the bowl"
    patcher = ReadoutPatcher(runner)
    return {
        "runner": runner,
        "patcher": patcher,
        "clean_batch": runner.prepare(clean, wrist, proprio, language),
        "perturbed_batch": runner.prepare(perturbed, wrist, proprio, language),
        "noise": runner.make_noise(0),
    }


def test_decode_matches_model_path(setup):
    patcher, runner = setup["patcher"], setup["runner"]
    direct = runner.chunk(dict(setup["clean_batch"]), setup["noise"])
    decoded = patcher.decode(patcher.prefill(setup["clean_batch"]), setup["noise"])
    assert np.allclose(direct, decoded, atol=1e-5)


def test_fully_clean_cache_reproduces_clean_action(setup):
    patcher = setup["patcher"]
    clean = patcher.prefill(setup["clean_batch"])
    perturbed = patcher.prefill(setup["perturbed_batch"])
    restored = patcher.hybrid(perturbed, clean, range(patcher.num_layers))
    assert np.allclose(
        patcher.decode(restored, setup["noise"]), patcher.decode(clean, setup["noise"]), atol=1e-5
    )


def test_empty_patch_keeps_perturbed_action(setup):
    patcher = setup["patcher"]
    clean = patcher.prefill(setup["clean_batch"])
    perturbed = patcher.prefill(setup["perturbed_batch"])
    same = patcher.hybrid(perturbed, clean, [])
    assert np.allclose(
        patcher.decode(same, setup["noise"]), patcher.decode(perturbed, setup["noise"]), atol=1e-5
    )


def test_first_layer_differs_only_in_agentview_tokens(setup):
    patcher = setup["patcher"]
    clean = patcher.prefill(setup["clean_batch"])
    perturbed = patcher.prefill(setup["perturbed_batch"])
    difference = (clean.values[0] - perturbed.values[0]).abs().sum(dim=(0, 1, 3))
    assert difference[AGENTVIEW_TOKENS].min() > 0
    assert difference[IMAGE_TOKENS:].max() == 0


def test_repeated_decode_is_stable(setup):
    patcher = setup["patcher"]
    clean = patcher.prefill(setup["clean_batch"])
    first = patcher.decode(clean, setup["noise"])
    second = patcher.decode(clean, setup["noise"])
    assert np.array_equal(first, second)


def test_clean_connector_output_reproduces_clean_action(setup):
    patcher = setup["patcher"]
    clean = patcher.capture(setup["clean_batch"], [])["connector"]
    patched = patcher.prefill_with_connector(setup["perturbed_batch"], clean)
    assert np.allclose(
        patcher.decode(patched, setup["noise"]),
        patcher.decode(patcher.prefill(setup["clean_batch"]), setup["noise"]),
        atol=1e-5,
    )
