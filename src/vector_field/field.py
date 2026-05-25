from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


VectorFunc = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class VectorField:
    """2D vector field visualized as a quiver plot."""

    def __init__(
        self,
        func: VectorFunc,
        x_range: tuple[float, float] = (-5.0, 5.0),
        y_range: tuple[float, float] = (-5.0, 5.0),
        density: int = 20,
    ) -> None:
        self.func = func
        self.x_range = x_range
        self.y_range = y_range
        self.density = density

    def _grid(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(*self.x_range, self.density)
        y = np.linspace(*self.y_range, self.density)
        return np.meshgrid(x, y)

    def plot(
        self,
        *,
        normalize: bool = False,
        color_by_magnitude: bool = True,
        ax: plt.Axes | None = None,
        title: str = "",
    ) -> Figure:
        X, Y = self._grid()
        U, V = self.func(X, Y)

        magnitude = np.sqrt(U**2 + V**2)

        if normalize:
            with np.errstate(invalid="ignore", divide="ignore"):
                U = np.where(magnitude > 0, U / magnitude, 0.0)
                V = np.where(magnitude > 0, V / magnitude, 0.0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        else:
            fig = ax.get_figure()

        if color_by_magnitude:
            ax.quiver(X, Y, U, V, magnitude, cmap="viridis", angles="xy", pivot="mid")
        else:
            ax.quiver(X, Y, U, V, angles="xy", pivot="mid")
        ax.set_xlim(*self.x_range)
        ax.set_ylim(*self.y_range)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if title:
            ax.set_title(title)

        return fig  # type: ignore[return-value]
