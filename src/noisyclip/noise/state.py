"""Sample noise-state records, protocols, and transactional JSON storage."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from noisyclip.data.records import Batch, SampleRecord
from noisyclip.models.outputs import ModelOutput

STATE_FORMAT_VERSION = 1
VALID_PARTITIONS = frozenset({"trusted", "uncertain", "suspicious"})


@dataclass(slots=True)
class SampleState:
    """Persistent per-sample trust state for noise-aware training.

    Scores and weights use the closed range `[0, 1]`. `partition` is one of
    `trusted`, `uncertain`, or `suspicious`. `pseudo_target` is an internal
    class index or `None` when no pseudo-label is active.
    """

    sample_id: str
    seen_count: int
    ema_loss: float
    ema_probs: list[float] | None
    prediction_stability: float
    augmentation_agreement: float
    prototype_similarity: float
    prototype_margin: float
    trust_score: float
    supervised_weight: float
    partition: str
    pseudo_target: int | None
    pseudo_confidence: float | None
    updated_epoch: int


class JsonSampleStateStore:
    """Persist sample states by stable `sample_id` with transactional commits.

    Args:
        root: Directory used for state manifests and epoch JSON files. The
            directory may be empty or contain a previous committed manifest.
        expected_sample_ids: Optional complete train/validation sample-id set.
            When provided, `commit_epoch` requires every expected id exactly
            once. IDs are strings and returned by `load` in caller order.

    Raises:
        ValueError: If `expected_sample_ids` contains duplicates or empty IDs.
        OSError: If state files cannot be read or written.
    """

    def __init__(self, root: Path | str, expected_sample_ids: Sequence[str] | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "manifest.json"
        self._staged_dir = self.root / ".staged"
        self._staged_dir.mkdir(exist_ok=True)
        self._expected_sample_ids = (
            _validate_unique_sample_ids(expected_sample_ids, field_name="expected_sample_ids")
            if expected_sample_ids is not None
            else None
        )

    @property
    def latest_epoch(self) -> int | None:
        """Return the last committed epoch, or `None` when no commit exists."""

        manifest = self._read_manifest(required=False)
        if manifest is None:
            return None
        return int(manifest["epoch"])

    def load(self, sample_ids: list[str]) -> list[SampleState]:
        """Load committed states for `sample_ids` in the requested order.

        Args:
            sample_ids: Non-empty stable IDs. Duplicates are rejected.

        Returns:
            List of `SampleState` objects ordered exactly like `sample_ids`.

        Raises:
            ValueError: If ids are duplicated, no committed epoch exists, a
                requested id is missing, or stored state validation fails.
            OSError: If the committed JSON file cannot be read.
        """

        requested = _validate_unique_sample_ids(sample_ids, field_name="sample_ids")
        manifest = self._read_manifest(required=True)
        if manifest is None:
            raise ValueError("No committed sample state is available.")
        states = self._read_state_file(
            self.root / str(manifest["state_file"]), int(manifest["epoch"])
        )
        by_id = {state.sample_id: state for state in states}
        missing = [sample_id for sample_id in requested if sample_id not in by_id]
        if missing:
            raise ValueError(f"Committed state is missing requested sample_id(s): {missing}.")
        return [by_id[sample_id] for sample_id in requested]

    def load_all(self) -> list[SampleState]:
        """Load all states from the latest committed epoch sorted by `sample_id`.

        Returns:
            All committed states in deterministic `sample_id` order.

        Raises:
            ValueError: If no committed state exists or the file is invalid.
            OSError: If the committed JSON file cannot be read.
        """

        manifest = self._read_manifest(required=True)
        if manifest is None:
            raise ValueError("No committed sample state is available.")
        states = self._read_state_file(
            self.root / str(manifest["state_file"]), int(manifest["epoch"])
        )
        return sorted(states, key=lambda state: state.sample_id)

    def stage_epoch(self, states: list[SampleState], epoch: int) -> Path:
        """Write a complete uncommitted epoch state file.

        Args:
            states: Complete per-sample states. Each `sample_id` must appear
                once; numeric scores must be finite and within their documented
                ranges; `updated_epoch` must equal `epoch`.
            epoch: Non-negative epoch number stored in the staged metadata.

        Returns:
            Path to the staged JSON file.

        Raises:
            ValueError: If ids, ranges, partitions, or epoch fields are invalid.
            OSError: If the temporary file cannot be written.
        """

        _validate_epoch(epoch)
        latest = self.latest_epoch
        if latest is not None and epoch <= latest:
            raise ValueError(f"Cannot stage epoch {epoch}; latest committed epoch is {latest}.")
        _validate_states(states, expected_epoch=epoch)
        payload = {
            "version": STATE_FORMAT_VERSION,
            "epoch": epoch,
            "states": [asdict(state) for state in states],
        }
        staged_path = self._staged_dir / f"epoch_{epoch:04d}.json"
        tmp_path = staged_path.with_suffix(".json.tmp")
        if staged_path.exists() or tmp_path.exists():
            raise ValueError(
                f"Uncommitted state already exists for epoch {epoch}; roll it back first."
            )
        _write_json_atomic(tmp_path, payload)
        os.replace(tmp_path, staged_path)
        return staged_path

    def commit_epoch(self, epoch: int) -> None:
        """Atomically publish the staged state for `epoch`.

        Args:
            epoch: Non-negative epoch number matching a staged file.

        Raises:
            ValueError: If no staged file exists, epoch/version metadata is
                inconsistent, sample ids are incomplete, or validation fails.
            OSError: If publishing the state or manifest fails.
        """

        _validate_epoch(epoch)
        latest = self.latest_epoch
        if latest is not None and epoch <= latest:
            raise ValueError(f"Cannot commit epoch {epoch}; latest committed epoch is {latest}.")
        staged_path = self._staged_dir / f"epoch_{epoch:04d}.json"
        if not staged_path.is_file():
            raise ValueError(f"No staged sample state exists for epoch {epoch}.")
        states = self._read_state_file(staged_path, epoch)
        if self._expected_sample_ids is not None:
            actual = {state.sample_id for state in states}
            expected = set(self._expected_sample_ids)
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            if missing or extra:
                raise ValueError(
                    "Staged state sample_id set does not match expected ids: "
                    f"missing={missing}, extra={extra}."
                )
        final_name = f"epoch_{epoch:04d}.json"
        final_path = self.root / final_name
        if final_path.exists():
            raise ValueError(f"Committed sample state already exists for epoch {epoch}.")
        os.link(staged_path, final_path)
        staged_path.unlink()
        manifest = {"version": STATE_FORMAT_VERSION, "epoch": epoch, "state_file": final_name}
        _write_json_atomic(self._manifest_path, manifest)

    def rollback_uncommitted(self) -> None:
        """Delete staged files while preserving every committed epoch.

        Raises:
            OSError: If an uncommitted file cannot be removed.
        """

        if not self._staged_dir.exists():
            return
        for path in self._staged_dir.glob("epoch_*.json"):
            path.unlink()
        for path in self._staged_dir.glob("epoch_*.json.tmp"):
            path.unlink()

    def _read_manifest(self, *, required: bool) -> dict[str, Any] | None:
        if not self._manifest_path.is_file():
            if required:
                raise ValueError("Sample state manifest is missing.")
            return None
        with self._manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Sample state manifest must be a JSON object.")
        if payload.get("version") != STATE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported sample state version: {payload.get('version')!r}; "
                f"expected {STATE_FORMAT_VERSION}."
            )
        epoch = payload.get("epoch")
        state_file = payload.get("state_file")
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Sample state manifest has an invalid epoch.")
        if not isinstance(state_file, str) or not state_file:
            raise ValueError("Sample state manifest has an invalid state_file.")
        if Path(state_file).name != state_file or state_file != f"epoch_{epoch:04d}.json":
            raise ValueError("Sample state manifest has an unsafe or inconsistent state_file.")
        return payload

    def _read_state_file(self, path: Path, expected_epoch: int) -> list[SampleState]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Sample state file must be a JSON object: {path}.")
        if payload.get("version") != STATE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported sample state version in {path}: {payload.get('version')!r}."
            )
        if payload.get("epoch") != expected_epoch:
            raise ValueError(
                f"Sample state epoch mismatch in {path}: "
                f"expected {expected_epoch}, got {payload.get('epoch')!r}."
            )
        raw_states = payload.get("states")
        if not isinstance(raw_states, list):
            raise ValueError(f"Sample state payload must contain a states list: {path}.")
        states = [_state_from_mapping(item) for item in raw_states]
        _validate_states(states, expected_epoch=expected_epoch)
        return states


class PrototypeBuilder(Protocol):
    """Protocol for building class prototypes from embeddings."""

    def fit(
        self,
        embeddings: Tensor,
        targets: Tensor,
        sample_weights: Tensor | None,
        num_classes: int,
    ) -> Tensor:
        """Return `[C, D]` L2-normalized prototypes or raise on missing classes."""


class TrustSignal(Protocol):
    """Protocol for computing one raw per-sample trust signal."""

    name: str

    def compute(
        self,
        batch: Batch,
        output_weak: ModelOutput,
        output_strong: ModelOutput | None,
        state: list[SampleState],
        prototypes: Tensor | None,
    ) -> Tensor:
        """Return raw `[B]` signal values without normalization or persistence."""


class TrustAggregator(Protocol):
    """Protocol for class-aware aggregation of trust signals."""

    def update_epoch(
        self,
        records: list[SampleRecord],
        raw_signals: Mapping[str, Tensor],
        previous: list[SampleState],
        epoch: int,
    ) -> list[SampleState]:
        """Return validated epoch states after class-wise normalization."""


class SampleStateStore(Protocol):
    """Protocol for transactional storage of per-sample state."""

    def load(self, sample_ids: list[str]) -> list[SampleState]:
        """Load states matching `sample_ids` in the requested order."""

    def stage_epoch(self, states: list[SampleState], epoch: int) -> Path:
        """Write uncommitted epoch state and return the staged path."""

    def commit_epoch(self, epoch: int) -> None:
        """Atomically publish the staged state for `epoch`."""

    def rollback_uncommitted(self) -> None:
        """Remove staged, uncommitted state without touching committed epochs."""


def _state_from_mapping(raw: object) -> SampleState:
    if not isinstance(raw, dict):
        raise ValueError("Each sample state entry must be a JSON object.")
    try:
        return SampleState(
            sample_id=str(raw["sample_id"]),
            seen_count=int(raw["seen_count"]),
            ema_loss=float(raw["ema_loss"]),
            ema_probs=(
                None if raw.get("ema_probs") is None else [float(item) for item in raw["ema_probs"]]
            ),
            prediction_stability=float(raw["prediction_stability"]),
            augmentation_agreement=float(raw["augmentation_agreement"]),
            prototype_similarity=float(raw["prototype_similarity"]),
            prototype_margin=float(raw["prototype_margin"]),
            trust_score=float(raw["trust_score"]),
            supervised_weight=float(raw["supervised_weight"]),
            partition=str(raw["partition"]),
            pseudo_target=(None if raw.get("pseudo_target") is None else int(raw["pseudo_target"])),
            pseudo_confidence=(
                None if raw.get("pseudo_confidence") is None else float(raw["pseudo_confidence"])
            ),
            updated_epoch=int(raw["updated_epoch"]),
        )
    except KeyError as exc:
        raise ValueError(f"Sample state entry is missing field {exc.args[0]!r}.") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Sample state entry has an invalid field type: {exc}.") from exc


def _validate_unique_sample_ids(
    sample_ids: Sequence[str] | None,
    *,
    field_name: str,
) -> list[str]:
    if sample_ids is None:
        raise ValueError(f"{field_name} must not be None.")
    ids = list(sample_ids)
    if not ids:
        raise ValueError(f"{field_name} must be non-empty.")
    seen: set[str] = set()
    duplicates: list[str] = []
    for sample_id in ids:
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{field_name} contains an empty or non-string sample_id.")
        if sample_id in seen:
            duplicates.append(sample_id)
        seen.add(sample_id)
    if duplicates:
        raise ValueError(f"{field_name} contains duplicate sample_id(s): {duplicates}.")
    return ids


def _validate_epoch(epoch: int) -> None:
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"epoch must be a non-negative integer, got {epoch!r}.")


def _validate_probability(value: float | None, *, field_name: str) -> None:
    if value is None:
        return
    tensor = torch.tensor(value, dtype=torch.float64)
    if not bool(torch.isfinite(tensor)):
        raise ValueError(f"{field_name} must be finite.")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {value}.")


def _validate_finite_nonnegative(value: float, *, field_name: str) -> None:
    tensor = torch.tensor(value, dtype=torch.float64)
    if not bool(torch.isfinite(tensor)):
        raise ValueError(f"{field_name} must be finite.")
    if value < 0.0:
        raise ValueError(f"{field_name} must be non-negative, got {value}.")


def _validate_states(states: list[SampleState], *, expected_epoch: int) -> None:
    _validate_unique_sample_ids([state.sample_id for state in states], field_name="states")
    for state in states:
        if state.seen_count < 0:
            raise ValueError(f"seen_count must be non-negative for sample_id={state.sample_id}.")
        if state.updated_epoch != expected_epoch:
            raise ValueError(
                f"updated_epoch must equal committed epoch for sample_id={state.sample_id}: "
                f"expected {expected_epoch}, got {state.updated_epoch}."
            )
        _validate_finite_nonnegative(state.ema_loss, field_name="ema_loss")
        if state.ema_probs is not None:
            if not state.ema_probs:
                raise ValueError(f"ema_probs must be non-empty for sample_id={state.sample_id}.")
            for index, value in enumerate(state.ema_probs):
                _validate_probability(value, field_name=f"ema_probs[{index}]")
            if abs(sum(state.ema_probs) - 1.0) > 1e-4:
                raise ValueError(f"ema_probs must sum to 1 for sample_id={state.sample_id}.")
        _validate_probability(state.prediction_stability, field_name="prediction_stability")
        _validate_probability(state.augmentation_agreement, field_name="augmentation_agreement")
        _validate_probability(state.prototype_similarity, field_name="prototype_similarity")
        _validate_probability(state.prototype_margin, field_name="prototype_margin")
        _validate_probability(state.trust_score, field_name="trust_score")
        _validate_probability(state.supervised_weight, field_name="supervised_weight")
        if state.partition not in VALID_PARTITIONS:
            raise ValueError(
                f"partition must be one of {sorted(VALID_PARTITIONS)} for "
                f"sample_id={state.sample_id}, got {state.partition!r}."
            )
        if state.pseudo_target is not None and state.pseudo_target < 0:
            raise ValueError(f"pseudo_target must be non-negative for sample_id={state.sample_id}.")
        _validate_probability(state.pseudo_confidence, field_name="pseudo_confidence")


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
