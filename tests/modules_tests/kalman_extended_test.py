"""Add a doc string to my files."""

import numpy as np

from ekf_slam_3d.data_classes.lie_algebra import SE3
from ekf_slam_3d.data_classes.sensors import (
    angle_mask,
    inverse_distance_azimuth_slam,
    measure_distance_azimuth_slam,
    measure_gps,
    step_dynamics,
    step_dynamics_slam,
)
from ekf_slam_3d.modules.kalman_extended import ExtendedKalmanFilter, MeasurementSpec
from ekf_slam_3d.modules.state_space import StateSpaceNonlinear


def _slam_ekf(pose_cov: np.ndarray, num_landmarks: int = 2) -> ExtendedKalmanFilter:
    cov = 100.0 * np.eye(6 + 2 * num_landmarks)
    cov[:6, :6] = pose_cov
    return ExtendedKalmanFilter(
        state_space_nonlinear=StateSpaceNonlinear(motion_model=step_dynamics_slam),
        initial_x=np.zeros((6 + 2 * num_landmarks, 1)),
        initial_covariance=cov,
        process_noise=1e-3 * np.eye(2),
        measurement_noise=1e-2,
    )


def test_initialize_from_measurement_propagates_pose_and_sensor_uncertainty() -> None:
    """Test that a seeded landmark gets the standard EKF-SLAM covariance.

    For a sighting (d, a) from pose (x, y, yaw), the landmark is
    (x + d cos(yaw + a), y + d sin(yaw + a)); its covariance must be
    G_pose P_pose G_pose^T + G_z R G_z^T, and its cross-covariance with the pose
    G_pose P_pose - not zero, or the map can't be corrected along with the pose.
    """
    # Arrange
    pose_cov = np.diag([0.1, 0.2, 0.0, 0.0, 0.0, 0.05])
    ekf = _slam_ekf(pose_cov)
    distance, azimuth = 5.0, 0.3
    z = np.array([[distance], [azimuth]])
    R = np.diag([0.01, 0.002])

    c, s = np.cos(azimuth), np.sin(azimuth)
    G_pose = np.zeros((2, 6))
    G_pose[:, 0] = [1.0, 0.0]
    G_pose[:, 1] = [0.0, 1.0]
    G_pose[:, 5] = [-distance * s, distance * c]
    G_z = np.array([[c, -distance * s], [s, distance * c]])

    # Act
    ekf.initialize_from_measurement(
        idx=8,
        z=z,
        inverse_model=inverse_distance_azimuth_slam,
        measurement_covariance=R,
    )

    # Assert
    np.testing.assert_array_almost_equal(ekf.x[8:10, 0], [distance * c, distance * s])
    np.testing.assert_array_almost_equal(
        ekf.cov[8:10, 8:10], G_pose @ pose_cov @ G_pose.T + G_z @ R @ G_z.T, decimal=5
    )
    np.testing.assert_array_almost_equal(
        ekf.cov[8:10, :6], G_pose @ pose_cov, decimal=5
    )
    np.testing.assert_array_almost_equal(ekf.cov[8:10, 6:8], np.zeros((2, 2)))
    np.testing.assert_array_almost_equal(ekf.cov, ekf.cov.T)


def test_update_uses_the_given_measurement_covariance() -> None:
    """Test that a tighter measurement covariance pulls the estimate harder."""
    # Arrange
    true_position = np.array([[5.0], [3.0], [0.0]])

    def make_ekf() -> ExtendedKalmanFilter:
        return ExtendedKalmanFilter(
            state_space_nonlinear=StateSpaceNonlinear(motion_model=step_dynamics),
            initial_x=np.zeros((6, 1)),
            initial_covariance=np.eye(6),
            process_noise=1e-3 * np.eye(2),
            measurement_noise=1.0,
        )

    loose, tight = make_ekf(), make_ekf()

    # Act
    loose.update(
        z=true_position, sensor=measure_gps, u=np.zeros((2, 1)), measurement_args=[]
    )
    tight.update(
        z=true_position,
        sensor=measure_gps,
        u=np.zeros((2, 1)),
        measurement_args=[],
        spec=MeasurementSpec(covariance=1e-4 * np.eye(3)),
    )

    # Assert
    np.testing.assert_array_almost_equal(loose.x[:3], true_position / 2)
    np.testing.assert_array_almost_equal(tight.x[:3], true_position, decimal=3)


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


def test_extended_kalman_filter_update_wraps_azimuth_across_branch_cut() -> None:
    """Test that a bearing straddling +-pi doesn't trigger a huge, spurious correction.

    A landmark sighting whose azimuth crosses the atan2 branch cut (e.g. true bearing
    just under pi, measured bearing just over -pi) has a small *actual* angular error
    but a raw (z - predict_z) difference near 2*pi. Without wrapping that innovation,
    the filter misreads it as a huge error and yaws the pose estimate wildly; with
    wrapping, the correction should stay proportional to the real, small bearing error.
    """
    # Arrange: a landmark placed just shy of the branch cut (true bearing = pi - 0.02)
    pose = SE3(xyz=np.array([0.0, 0.0, 0.0]))
    distance = 10.0
    true_bearing = np.pi - 0.02
    landmark = np.array(
        [[distance * np.cos(true_bearing)], [distance * np.sin(true_bearing)]]
    )
    initial_x = np.vstack((pose.as_vector(), landmark))
    initial_cov = 0.05 * np.eye(8)

    def make_ekf() -> ExtendedKalmanFilter:
        return ExtendedKalmanFilter(
            state_space_nonlinear=StateSpaceNonlinear(motion_model=step_dynamics_slam),
            initial_x=initial_x.copy(),
            initial_covariance=initial_cov.copy(),
            process_noise=1e-3 * np.eye(2),
            measurement_noise=1e-2,
        )

    ekf = make_ekf()
    predict_z = measure_distance_azimuth_slam(ekf.x, args=([0], 1))

    # measured azimuth crossed to the other side of the branch cut - the real bearing
    # error is only ~0.04 rad, not the ~2*pi implied by the raw difference
    measured_azimuth = -np.pi + 0.02
    z = np.array([[predict_z[0, 0]], [measured_azimuth]])
    mask = angle_mask(len(z), stride=2, offsets=(1,))
    u = np.zeros((2, 1))

    def unregistered_sensor(state: np.ndarray, args: tuple) -> np.ndarray:
        return measure_distance_azimuth_slam(state, args)

    # Act
    ekf_default = make_ekf()
    ekf_explicit = make_ekf()
    ekf_unwrapped = make_ekf()
    ekf_default.update(
        z=z, sensor=measure_distance_azimuth_slam, u=u, measurement_args=([0], 1)
    )
    ekf_explicit.update(
        z=z,
        sensor=unregistered_sensor,
        u=u,
        measurement_args=([0], 1),
        spec=MeasurementSpec(angle_mask=mask),
    )
    ekf_unwrapped.update(
        z=z, sensor=unregistered_sensor, u=u, measurement_args=([0], 1)
    )

    # Assert: registered sensors wrap by default; an unregistered one needs the mask
    assert abs(ekf_default.x[5, 0]) < 0.5
    assert abs(ekf_explicit.x[5, 0]) < 0.5
    assert abs(ekf_unwrapped.x[5, 0]) > 1.0
