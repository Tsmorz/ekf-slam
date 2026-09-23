"""Add a doc string to my files."""

import numpy as np
import pytest

from config.definitions import COVARIANCE_ELLIPSE_SIGMA, VIEW_MIN_MARGIN
from ekf_slam_3d.data_classes.lie_algebra import SE3
from ekf_slam_3d.data_classes.map import Map
from ekf_slam_3d.data_classes.sensors import step_dynamics
from ekf_slam_3d.modules.simulators import (
    SlamSimulator,
    ellipse_geometry,
    expand_view_bounds,
)
from ekf_slam_3d.modules.state_space import StateSpaceNonlinear


def test_expand_view_bounds_starts_around_the_points() -> None:
    """Test that the first view contains every point with at least the minimum margin."""
    # Act
    bounds = expand_view_bounds(None, xs=[0.0, 1.0], ys=[0.0, 1.0])

    # Assert
    assert bounds == pytest.approx(
        (
            -VIEW_MIN_MARGIN,
            1.0 + VIEW_MIN_MARGIN,
            -VIEW_MIN_MARGIN,
            1.0 + VIEW_MIN_MARGIN,
        )
    )


def test_expand_view_bounds_leaves_the_view_alone_when_everything_fits() -> None:
    """Test that points already in view don't move the view."""
    # Act / Assert
    assert expand_view_bounds((0.0, 10.0, 0.0, 10.0), xs=[1.0, 9.0], ys=[5.0]) is None


def test_expand_view_bounds_only_grows_the_side_that_was_exceeded() -> None:
    """Test that the view grows outward on the violated side and never pans or shrinks."""
    # Act
    bounds = expand_view_bounds((0.0, 10.0, 0.0, 10.0), xs=[12.0], ys=[5.0])

    # Assert
    assert bounds is not None
    x_min, x_max, y_min, y_max = bounds
    assert (x_min, y_min, y_max) == (0.0, 0.0, 10.0)
    assert x_max > 12.0


def test_ellipse_geometry_of_a_rotated_covariance() -> None:
    """Test the ellipse axes and orientation for a covariance rotated by 30 degrees."""
    # Arrange
    angle = np.radians(30.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    covariance = rotation @ np.diag([0.25, 4.0]) @ rotation.T

    # Act
    width, height, degrees = ellipse_geometry(covariance)

    # Assert
    assert width == pytest.approx(2 * COVARIANCE_ELLIPSE_SIGMA * 0.5)
    assert height == pytest.approx(2 * COVARIANCE_ELLIPSE_SIGMA * 2.0)
    assert np.tan(np.radians(degrees)) == pytest.approx(np.tan(angle))


def test_simulator_process_noise_is_a_covariance() -> None:
    """Test that the simulator injects noise whose variance is the given covariance."""
    # Arrange
    np.random.seed(0)
    sim = SlamSimulator(
        state_space_nl=StateSpaceNonlinear(motion_model=step_dynamics),
        process_noise=np.diag([0.04, 0.0]),
        initial_pose=SE3(),
        sim_map=Map(),
    )

    # Act: with no turning, the distance driven each step is 1 + velocity noise
    steps = [sim.step(u=np.array([[1.0], [0.0]])).x for _ in range(5000)]

    # Assert
    assert np.var(np.diff(steps)) == pytest.approx(0.04, rel=0.1)
