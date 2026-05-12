"""
JAX runtime guard for the analytic pricer kernels.

The Fourier (COS, Carr-Madan), MGF, and implied-vol pricers rely on
``complex128`` transcendental ops (Heston Riccati, COS payoff coeffs, FFT).
JAX silently truncates ``complex128`` to ``complex64`` when ``jax_enable_x64``
is off, which produces catastrophically wrong prices for jump-laden models
like Bates while the per-call result still *looks* like a valid number.

On Apple Silicon the default JAX backend is Metal/MPS, which has no float64
hardware. Enabling ``jax_enable_x64`` after Metal is initialised segfaults
on the first complex op, so we cannot fix this from inside the pricer.
The user has to start the process with::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python ...

(or equivalent in code, before any JAX backend init).

This module turns the silent failure into a loud, single error at the first
pricer call so the wrong-answer trap is impossible.
"""

from __future__ import annotations

import functools
import os
from typing import Callable, TypeVar

import jax

F = TypeVar("F", bound=Callable[..., object])


def _diagnose_unsafe_runtime() -> str | None:
    """Return a remediation message if the JAX runtime is unsafe for pricing,
    else ``None``. Unsafe = ``jax_enable_x64`` is False, because the JIT'd
    kernels will then truncate to ``complex64`` and silently mis-price."""
    if jax.config.jax_enable_x64:
        return None

    backend = jax.default_backend()
    return (
        f"JAX pricer requires float64/complex128 arithmetic, but "
        f"jax_enable_x64 is False (default backend: {backend}). "
        f"On Apple Silicon, JAX defaults to Metal/MPS which lacks float64 "
        f"hardware; enabling x64 in the same process segfaults. "
        f"Start the process with these env vars BEFORE importing JAX:\n"
        f"    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1\n"
        f"or, for an existing CPU-default process, "
        f"`jax.config.update('jax_enable_x64', True)` before importing this module. "
        f"For Bates/Heston/Merton on Mac without changing env, use the pure-NumPy "
        f"pricers in pricer.bates_mgf_pricer / heston_mgf_pricer / merton_mgf_pricer "
        f"instead -- they are correct and need no JAX."
    )


def require_safe_runtime() -> None:
    """Raise RuntimeError if the JAX runtime cannot price safely."""
    msg = _diagnose_unsafe_runtime()
    if msg is not None:
        raise RuntimeError(msg)


def guard_pricer(fn: F) -> F:
    """Decorator: raise on first call if the JAX runtime is unsafe for pricing.

    Use to wrap each public JAX pricer entry point. The check is per-call
    (not per-import) so module import never fails, and so users who repair
    their environment between import and first call still succeed.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        require_safe_runtime()
        return fn(*args, **kwargs)

    return wrapper


# Allow opting out of the guard in tests that intentionally exercise the
# unsafe path. Set OPTIONS_DESK_DISABLE_PRICER_GUARD=1 to bypass.
if os.environ.get("OPTIONS_DESK_DISABLE_PRICER_GUARD") == "1":
    def guard_pricer(fn: F) -> F:  # type: ignore[no-redef]
        return fn

    def require_safe_runtime() -> None:  # type: ignore[no-redef]
        return None
