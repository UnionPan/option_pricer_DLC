"""
JAX backend policy for process simulation.

Backend resolution order:
  1. ``JAX_PLATFORMS`` if the user set one explicitly
  2. ``OPTIONS_DESK_JAX_BACKEND`` project-level override
  3. Auto-detect: CUDA GPU → Apple Metal (MPS) → CPU

Precision modes:
  - ``high``: float64 / complex128 (CPU, CUDA) — full COS accuracy
  - ``metal_safe``: float32 / complex64 (Apple MPS) — reduced precision,
    validated for typical equity Heston parameters

The precision mode is auto-detected from the active backend but can be
overridden via ``OPTIONS_DESK_JAX_PRECISION=high|metal_safe``.
"""

from __future__ import annotations

import os
from typing import Literal

import jax
import jax.numpy as jnp


_PROJECT_BACKEND_ENV = "OPTIONS_DESK_JAX_BACKEND"
_STRICT_BACKEND_ENV = "OPTIONS_DESK_JAX_STRICT_BACKEND"
_PRECISION_ENV = "OPTIONS_DESK_JAX_PRECISION"
_CONFIGURED = False

PrecisionMode = Literal["high", "metal_safe"]


def normalize_backend_name(name: str | None) -> str | None:
    """Normalize project backend aliases to JAX platform names."""
    if name is None:
        return None

    normalized = str(name).strip().lower()
    if normalized in ("", "auto", "default"):
        return None
    if normalized == "metal":
        return "mps"
    return normalized


def _detect_gpu_backend() -> str:
    """Try CUDA GPU → Apple Metal (MPS) → CPU.

    All three are supported. Metal runs in ``metal_safe`` precision mode
    (float32/complex64) because it lacks float64 hardware support.

    Skips CUDA probe on macOS (no CUDA on Apple Silicon) to avoid
    noisy "Failed to create Metal device" stderr from the GPU backend.
    """
    import platform
    import contextlib, io, sys

    # CUDA first (full float64 support) — skip on macOS
    if platform.system() != "Darwin":
        try:
            gpu_devices = jax.devices("gpu")
            if gpu_devices:
                return "gpu"
        except (RuntimeError, ValueError):
            pass

    # Apple Metal second (float32 only, but still faster than CPU for batched work)
    # Suppress native stderr during MPS probe to avoid "Failed to create MPS client" noise
    try:
        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        _old_stderr_fd = os.dup(2)
        os.dup2(_devnull_fd, 2)
        try:
            mps_devices = jax.devices("mps")
        finally:
            os.dup2(_old_stderr_fd, 2)
            os.close(_old_stderr_fd)
            os.close(_devnull_fd)
        if mps_devices:
            return "mps"
    except (RuntimeError, ValueError):
        pass
    return "cpu"


def get_backend_preference() -> str:
    """
    Resolve the backend preference for this process runtime.

    Precedence:
      1. JAX_PLATFORMS if already set by the caller
      2. OPTIONS_DESK_JAX_BACKEND
      3. Auto-detect: CUDA GPU if available, else CPU
    """
    jax_platforms = os.getenv("JAX_PLATFORMS")
    if jax_platforms:
        return jax_platforms

    project_backend = normalize_backend_name(os.getenv(_PROJECT_BACKEND_ENV))
    if project_backend:
        return project_backend

    return _detect_gpu_backend()


def strict_backend_requested() -> bool:
    """Whether backend initialization failures should be raised instead of masked."""
    value = os.getenv(_STRICT_BACKEND_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def is_metal_backend() -> bool:
    """Return True if the active JAX backend is Apple Metal (MPS)."""
    try:
        return jax.default_backend() == "mps"
    except Exception:
        return False


def get_precision_mode() -> PrecisionMode:
    """Resolve the precision mode for Fourier/COS pricers.

    Precedence:
      1. ``OPTIONS_DESK_JAX_PRECISION`` env var (explicit override)
      2. Auto-detect from backend: MPS → ``metal_safe``, else ``high``
    """
    explicit = os.getenv(_PRECISION_ENV, "").strip().lower()
    if explicit in ("high", "metal_safe"):
        return explicit  # type: ignore[return-value]
    return "metal_safe" if is_metal_backend() else "high"


def get_float_dtype():
    """Return the appropriate float dtype for the current precision mode."""
    return jnp.float32 if get_precision_mode() == "metal_safe" else jnp.float64


def get_complex_dtype():
    """Return the appropriate complex dtype for the current precision mode."""
    return jnp.complex64 if get_precision_mode() == "metal_safe" else jnp.complex128


def configure_jax_runtime() -> str:
    """
    Configure JAX platforms and precision before the first backend operation.

    On CPU/CUDA: enables float64 for full COS pricing accuracy.
    On Metal: keeps float32 (Metal lacks float64 hardware).

    If JAX has already initialized its backend, platform selection becomes
    a no-op but precision detection still runs.
    """
    global _CONFIGURED
    backend = get_backend_preference()

    if _CONFIGURED:
        return backend

    try:
        jax.config.update("jax_platforms", backend)
    except Exception:
        # If JAX is already initialized, platform selection cannot be changed.
        pass

    # Enable float64 only on backends that support it
    if not is_metal_backend() and backend not in ("mps", "metal"):
        try:
            jax.config.update("jax_enable_x64", True)
        except Exception:
            pass

    _CONFIGURED = True
    return backend


def is_backend_init_error(exc: BaseException) -> bool:
    """Heuristic detection for runtime backend initialization failures."""
    message = str(exc)
    markers = (
        "Unable to initialize backend",
        "Failed to create MPS client",
        "No supported GPU was found",
        "Failed to create Metal device",
        "VisibleDeviceCount()",
        "UNKNOWN: -:0:0: error: unknown attribute code",
    )
    return any(marker in message for marker in markers)


def should_fallback_to_numpy(exc: BaseException) -> bool:
    """Return True when process simulation should gracefully fall back to NumPy."""
    return is_backend_init_error(exc) and not strict_backend_requested()
