"""Context-local control for redundant value checks in trusted hot paths."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_VALUE_CHECKS_ENABLED: ContextVar[bool] = ContextVar("value_checks_enabled", default=True)


def value_checks_enabled() -> bool:
    """Return whether tensor-value checks should synchronize the current device."""

    return _VALUE_CHECKS_ENABLED.get()


@contextmanager
def tensor_value_checks(*, enabled: bool) -> Iterator[None]:
    """Temporarily enable or suppress redundant finite/range checks."""

    token = _VALUE_CHECKS_ENABLED.set(enabled)
    try:
        yield
    finally:
        _VALUE_CHECKS_ENABLED.reset(token)
