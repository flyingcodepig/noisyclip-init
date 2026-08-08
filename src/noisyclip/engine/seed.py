"""Explicit RNG seeding, deterministic mode, and full RNG snapshots."""

from __future__ import annotations

import random
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass(slots=True)
class RngSnapshot:
    """Serializable Python, NumPy, and PyTorch RNG state.

    Attributes:
        python_state: State returned by `random.getstate()`.
        numpy_state: State tuple returned by `numpy.random.get_state()`.
        torch_cpu_state: Byte tensor for CPU RNG.
        torch_cuda_states: Per-device CUDA RNG tensors. Empty on CPU-only hosts.
        deterministic_algorithms: Whether deterministic torch algorithms were
            enabled at snapshot time.
    """

    python_state: object
    numpy_state: tuple[Any, ...]
    torch_cpu_state: Tensor
    torch_cuda_states: list[Tensor]
    deterministic_algorithms: bool

    def state_dict(self) -> dict[str, object]:
        """Return a checkpoint-safe mapping of RNG state tensors and metadata."""

        return {
            "python_state": self.python_state,
            "numpy_state": self.numpy_state,
            "torch_cpu_state": self.torch_cpu_state.detach().cpu().clone(),
            "torch_cuda_states": [state.detach().cpu().clone() for state in self.torch_cuda_states],
            "deterministic_algorithms": self.deterministic_algorithms,
        }

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, object]) -> RngSnapshot:
        """Restore an `RngSnapshot` from a checkpoint mapping.

        Args:
            state_dict: Mapping produced by `state_dict`.

        Returns:
            `RngSnapshot` with CPU tensors.

        Raises:
            TypeError: If tensor fields are malformed.
        """

        cpu_state = state_dict.get("torch_cpu_state")
        if not isinstance(cpu_state, Tensor):
            raise TypeError("RNG torch_cpu_state must be a tensor.")
        raw_cuda_states = state_dict.get("torch_cuda_states", [])
        if not isinstance(raw_cuda_states, list) or not all(
            isinstance(state, Tensor) for state in raw_cuda_states
        ):
            raise TypeError("RNG torch_cuda_states must be a list of tensors.")
        numpy_state = state_dict.get("numpy_state")
        if not isinstance(numpy_state, tuple):
            raise TypeError("RNG numpy_state must be a tuple.")
        return cls(
            python_state=state_dict.get("python_state"),
            numpy_state=tuple(numpy_state),
            torch_cpu_state=cpu_state.detach().cpu().clone(),
            torch_cuda_states=[state.detach().cpu().clone() for state in raw_cuda_states],
            deterministic_algorithms=bool(state_dict.get("deterministic_algorithms", False)),
        )


class SeedContext(AbstractContextManager["SeedContext"]):
    """Context manager that sets RNG state and restores it on exit.

    Args:
        seed: Non-negative seed applied to Python, NumPy, torch CPU, and CUDA.
        deterministic: Whether to enable deterministic torch algorithms.

    Raises:
        ValueError: If `seed` is negative.
    """

    def __init__(self, seed: int, *, deterministic: bool = False) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative.")
        self.seed = seed
        self.deterministic = deterministic
        self._previous: RngSnapshot | None = None

    def __enter__(self) -> SeedContext:
        """Snapshot current RNG state, then apply the requested seed."""

        self._previous = capture_rng_state()
        set_seed(self.seed, deterministic=self.deterministic)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Restore the RNG snapshot captured on entry."""

        if self._previous is not None:
            restore_rng_state(self._previous)
        return None


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Set Python, NumPy, and PyTorch RNGs without import-time side effects.

    Args:
        seed: Non-negative integer seed.
        deterministic: Strictly enables deterministic PyTorch algorithms and
            disables CUDNN benchmarking when true. Unsupported nondeterministic
            operations fail instead of emitting warnings.

    Raises:
        ValueError: If seed is negative.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative.")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def capture_rng_state() -> RngSnapshot:
    """Capture Python, NumPy, torch CPU, and CUDA RNG states.

    Returns:
        `RngSnapshot` containing CPU tensors for checkpoint serialization.
    """

    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return RngSnapshot(
        python_state=random.getstate(),
        numpy_state=tuple(np.random.get_state()),
        torch_cpu_state=torch.random.get_rng_state(),
        torch_cuda_states=[state.detach().cpu().clone() for state in cuda_states],
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )


def restore_rng_state(snapshot: RngSnapshot | Mapping[str, object]) -> None:
    """Restore a previously captured RNG snapshot.

    Args:
        snapshot: `RngSnapshot` or mapping produced by `RngSnapshot.state_dict`.

    Raises:
        TypeError: If mapping fields are malformed.
    """

    state = snapshot if isinstance(snapshot, RngSnapshot) else RngSnapshot.from_state_dict(snapshot)
    random.setstate(state.python_state)  # type: ignore[arg-type]
    np.random.set_state(state.numpy_state)
    torch.random.set_rng_state(state.torch_cpu_state)
    if torch.cuda.is_available() and state.torch_cuda_states:
        torch.cuda.set_rng_state_all(state.torch_cuda_states)
    torch.use_deterministic_algorithms(state.deterministic_algorithms, warn_only=False)
