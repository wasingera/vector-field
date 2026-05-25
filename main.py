import matplotlib.pyplot as plt

plt.style.use("dark_background")
import numpy as np

from vector_field import VectorField


def rotation(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return -y, x


def saddle(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x, -y


def radial(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x, y


def spiral(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return -y + 0.3 * x, x + 0.3 * y


fields = [
    (rotation, "Rotation"),
    (saddle,   "Saddle"),
    (radial,   "Radial"),
    (spiral,   "Spiral"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

for ax, (func, title) in zip(axes.flat, fields):
    VectorField(func, density=18).plot(normalize=True, ax=ax, title=title)

fig.suptitle("2D Vector Field Demo", fontsize=16, y=1.01)
fig.tight_layout()
fig.canvas.manager.toolbar.hide()  # type: ignore[union-attr]
plt.show()
