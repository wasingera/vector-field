import numpy as np
import pytest

from vector_field import VectorField


def rotation(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return -y, x


def test_plot_returns_figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vf = VectorField(rotation)
    fig = vf.plot()
    assert fig is not None
    plt.close(fig)


def test_plot_normalize() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vf = VectorField(rotation)
    fig = vf.plot(normalize=True)
    plt.close(fig)


def test_custom_range_and_density() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vf = VectorField(rotation, x_range=(-2.0, 2.0), y_range=(-2.0, 2.0), density=10)
    fig = vf.plot()
    plt.close(fig)


def test_accepts_external_axes() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    vf = VectorField(rotation)
    returned = vf.plot(ax=ax)
    assert returned is fig
    plt.close(fig)
