"""
Shared structured-function input helpers for JAX-compatible process models.

The convention is:
- use an immutable structured curve/surface when a time- or state-dependent
  function must be both serializable and JAX-traceable
- fall back to opaque Python callables only when structured data is unavailable
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import interpolate


def _as_1d_float64(name: str, values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return arr


def _validate_strictly_increasing(name: str, grid: np.ndarray) -> None:
    if len(grid) < 2:
        raise ValueError(f"{name} must contain at least two points")
    if not np.all(np.diff(grid) > 0):
        raise ValueError(f"{name} must be strictly increasing")


@dataclass(frozen=True)
class StructuredTimeCurve:
    times: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        times = _as_1d_float64("times", self.times)
        values = _as_1d_float64("values", self.values)
        if len(times) != len(values):
            raise ValueError("times and values must have the same length")
        _validate_strictly_increasing("times", times)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)

    @classmethod
    def constant(cls, value: float, t_min: float = 0.0, t_max: float = 1.0) -> "StructuredTimeCurve":
        return cls(
            times=np.array([t_min, t_max], dtype=np.float64),
            values=np.array([value, value], dtype=np.float64),
        )

    def as_numpy_callable(self) -> Callable[[float], float]:
        def _curve(t: float) -> float:
            return float(
                np.interp(
                    t,
                    self.times,
                    self.values,
                    left=self.values[0],
                    right=self.values[-1],
                )
            )

        return _curve


@dataclass(frozen=True)
class StructuredGridSurface:
    x_grid: np.ndarray
    y_grid: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        x_grid = _as_1d_float64("x_grid", self.x_grid)
        y_grid = _as_1d_float64("y_grid", self.y_grid)
        values = np.asarray(self.values, dtype=np.float64)
        if values.shape != (len(y_grid), len(x_grid)):
            raise ValueError("values must have shape (len(y_grid), len(x_grid))")
        _validate_strictly_increasing("x_grid", x_grid)
        _validate_strictly_increasing("y_grid", y_grid)
        object.__setattr__(self, "x_grid", x_grid)
        object.__setattr__(self, "y_grid", y_grid)
        object.__setattr__(self, "values", values)

    def as_numpy_callable(self) -> Callable[[np.ndarray, float], np.ndarray]:
        grid_interp = interpolate.RegularGridInterpolator(
            (self.y_grid, self.x_grid),
            self.values,
            bounds_error=False,
            fill_value=None,
        )

        def _surface(x: np.ndarray, y: float) -> np.ndarray:
            x_arr = np.asarray(x, dtype=np.float64)
            x_flat = x_arr.reshape(-1)
            y_clipped = np.clip(float(y), self.y_grid[0], self.y_grid[-1])
            x_clipped = np.clip(x_flat, self.x_grid[0], self.x_grid[-1])
            pts = np.column_stack([np.full_like(x_clipped, y_clipped), x_clipped])
            out = np.asarray(grid_interp(pts), dtype=np.float64)
            return out.reshape(x_arr.shape)

        return _surface
