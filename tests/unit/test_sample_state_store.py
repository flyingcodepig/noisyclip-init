"""Unit tests for sample-id keyed transactional state storage."""

from __future__ import annotations

import pytest

from noisyclip.noise.state import JsonSampleStateStore, SampleState


def _state(sample_id: str, *, epoch: int, partition: str = "trusted") -> SampleState:
    return SampleState(
        sample_id=sample_id,
        seen_count=1,
        ema_loss=0.2,
        ema_probs=[0.8, 0.2],
        prediction_stability=0.8,
        augmentation_agreement=0.9,
        prototype_similarity=0.7,
        prototype_margin=0.6,
        trust_score=0.75,
        supervised_weight=0.75,
        partition=partition,
        pseudo_target=None,
        pseudo_confidence=None,
        updated_epoch=epoch,
    )


def test_load_returns_requested_sample_id_order_after_shuffle(tmp_path) -> None:
    """Loading is keyed by sample_id, not DataLoader order."""

    store = JsonSampleStateStore(tmp_path, expected_sample_ids=["a", "b", "c"])
    store.stage_epoch([_state("a", epoch=0), _state("b", epoch=0), _state("c", epoch=0)], 0)
    store.commit_epoch(0)

    loaded = store.load(["c", "a", "b"])

    assert [state.sample_id for state in loaded] == ["c", "a", "b"]


def test_duplicate_or_missing_sample_ids_fail(tmp_path) -> None:
    """Duplicate staged ids and missing requested ids are rejected."""

    store = JsonSampleStateStore(tmp_path, expected_sample_ids=["a", "b"])

    with pytest.raises(ValueError, match="duplicate"):
        store.stage_epoch([_state("a", epoch=0), _state("a", epoch=0)], 0)

    store.stage_epoch([_state("a", epoch=0), _state("b", epoch=0)], 0)
    store.commit_epoch(0)

    with pytest.raises(ValueError, match="missing requested"):
        store.load(["a", "c"])


def test_commit_requires_complete_expected_id_set(tmp_path) -> None:
    """A staged epoch cannot commit when expected ids are missing."""

    store = JsonSampleStateStore(tmp_path, expected_sample_ids=["a", "b"])
    store.stage_epoch([_state("a", epoch=0)], 0)

    with pytest.raises(ValueError, match="missing"):
        store.commit_epoch(0)


def test_rollback_uncommitted_preserves_last_committed_state(tmp_path) -> None:
    """Rollback removes staged state without deleting committed epoch data."""

    store = JsonSampleStateStore(tmp_path, expected_sample_ids=["a"])
    store.stage_epoch([_state("a", epoch=0)], 0)
    store.commit_epoch(0)
    store.stage_epoch([_state("a", epoch=1, partition="suspicious")], 1)

    store.rollback_uncommitted()

    assert store.latest_epoch == 0
    assert store.load(["a"])[0].partition == "trusted"
    with pytest.raises(ValueError, match="No staged"):
        store.commit_epoch(1)


def test_resume_reads_last_committed_epoch_and_validates_state_ranges(tmp_path) -> None:
    """A new store instance recovers the latest committed epoch and rejects bad ranges."""

    store = JsonSampleStateStore(tmp_path, expected_sample_ids=["a"])
    store.stage_epoch([_state("a", epoch=2)], 2)
    store.commit_epoch(2)

    resumed = JsonSampleStateStore(tmp_path, expected_sample_ids=["a"])

    assert resumed.latest_epoch == 2
    assert resumed.load(["a"])[0].updated_epoch == 2

    with pytest.raises(ValueError, match="partition"):
        resumed.stage_epoch([_state("a", epoch=3, partition="clean")], 3)
