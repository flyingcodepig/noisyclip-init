"""Unit tests for deterministic RNG seeding and restoration."""

from __future__ import annotations

import random

import numpy as np
import torch

from noisyclip.engine.seed import SeedContext, capture_rng_state, restore_rng_state, set_seed


def test_seed_context_restores_python_numpy_and_torch_rng() -> None:
    """Leaving `SeedContext` restores the caller's RNG streams."""

    set_seed(123, deterministic=True)
    before = capture_rng_state()
    with SeedContext(999, deterministic=True):
        assert random.random() == random.Random(999).random()
        assert np.random.rand() == np.random.RandomState(999).rand()
        assert torch.rand(1).numel() == 1
    after_random = random.random()
    restore_rng_state(before)
    assert after_random == random.random()


def test_rng_snapshot_round_trip_reproduces_next_torch_value() -> None:
    """A serialized snapshot can reproduce the next sampled torch value."""

    set_seed(77)
    snapshot = capture_rng_state().state_dict()
    expected = torch.rand(3)
    restore_rng_state(snapshot)
    actual = torch.rand(3)
    assert torch.equal(actual, expected)


def test_deterministic_seed_and_restore_use_strict_mode() -> None:
    """Configured determinism fails on unsupported operations instead of warning."""

    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        set_seed(19, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()

        snapshot = capture_rng_state()
        torch.use_deterministic_algorithms(True, warn_only=True)
        assert torch.is_deterministic_algorithms_warn_only_enabled()

        restore_rng_state(snapshot)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn_only)
