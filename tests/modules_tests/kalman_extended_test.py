"""Add a doc string to my files."""

import numpy as np

from ekf_slam_3d.data_classes.sensors import measure_gps, step_dynamics
from ekf_slam_3d.modules.kalman_extended import ExtendedKalmanFilter
from ekf_slam_3d.modules.state_space import StateSpaceNonlinear


def test_extended_kalman_filter_predict_moves_state_forward() -> None:
    """Test that predict() advances the state using the nonlinear motion model."""
    # Arrange
    ekf = ExtendedKalmanFilter(
        state_space_nonlinear=StateSpaceNonlinear(motion_model=step_dynamics),
        initial_x=np.zeros((6, 1)),
        initial_covariance=np.eye(6),
        process_noise=1e-3 * np.eye(2),
        measurement_noise=1e-2,
    )
    control = np.array([[1.0], [0.0]])

    # Act
    ekf.predict(u=control)

    # Assert
    np.testing.assert_array_almost_equal(ekf.x[:2, 0], np.array([1.0, 0.0]))
    assert ekf.cov.shape == (6, 6)
    np.testing.assert_array_almost_equal(ekf.cov, ekf.cov.T)


def test_extended_kalman_filter_update_pulls_state_toward_measurement() -> None:
    """Test that update() corrects the state estimate toward a GPS measurement."""
    # Arrange
    initial_x = np.zeros((6, 1))
    ekf = ExtendedKalmanFilter(
        state_space_nonlinear=StateSpaceNonlinear(motion_model=step_dynamics),
        initial_x=initial_x,
        initial_covariance=10.0 * np.eye(6),
        process_noise=1e-3 * np.eye(2),
        measurement_noise=1e-2,
    )
    true_position = np.array([[5.0], [3.0], [0.0]])

    # Act
    ekf.update(
        z=true_position, sensor=measure_gps, u=np.zeros((2, 1)), measurement_args=[]
    )

    # Assert
    before = np.linalg.norm(initial_x[:3] - true_position)
    after = np.linalg.norm(ekf.x[:3] - true_position)
    assert after < before
