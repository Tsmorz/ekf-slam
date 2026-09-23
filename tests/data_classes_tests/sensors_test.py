"""Add a doc string to my files."""

import numpy as np
import pytest

from ekf_slam_3d.data_classes.lie_algebra import SE3
from ekf_slam_3d.data_classes.map import Feature
from ekf_slam_3d.data_classes.sensors import (
    angle_mask,
    distance_azimuth_covariance,
    features_in_range,
    initialize_landmark_estimate,
    inverse_distance_azimuth_map,
    inverse_distance_azimuth_slam,
    measure_distance_azimuth,
    measure_distance_azimuth_map,
    measure_distance_azimuth_slam,
    step_dynamics_map,
    step_dynamics_slam,
)


def test_angle_mask_marks_interleaved_angles() -> None:
    """Test that the mask flags the azimuth/elevation slots of each group."""
    # Act
    mask = angle_mask(length=6, stride=3, offsets=(1, 2))

    # Assert
    np.testing.assert_array_equal(mask, [False, True, True, False, True, True])


def test_distance_azimuth_covariance_matches_simulated_noise() -> None:
    """Test that the covariance matches the noise measure_distance_azimuth injects."""
    # Arrange
    np.random.seed(0)
    pose = SE3()
    feature = Feature(id=0, x=4.0, y=0.0)
    readings = np.hstack(
        [
            measure_distance_azimuth(pose.as_vector(), [feature], noise=0.1)
            for _ in range(20000)
        ]
    )

    # Act
    covariance = distance_azimuth_covariance(np.array([[4.0], [0.0]]), noise=0.1)

    # Assert
    np.testing.assert_allclose(np.var(readings, axis=1), np.diag(covariance), rtol=0.05)


def test_features_in_range() -> None:
    """Test that only features within range are reported visible."""
    # Arrange
    features = [Feature(id=0, x=3.0, y=4.0), Feature(id=1, x=6.0, y=8.0)]

    # Act / Assert
    assert features_in_range(SE3(), features, max_range=5.0) == [0]
    assert features_in_range(SE3(), features, max_range=10.0) == [0, 1]


def test_inverse_models_invert_a_sighting() -> None:
    """Test that both inverse models recover the sighted landmark's position."""
    # Arrange
    pose = SE3(xyz=np.array([1.0, 2.0, 0.0]), roll_pitch_yaw=np.array([0.0, 0.0, 0.5]))
    feature = Feature(id=0, x=6.0, y=-3.0)
    sighting = measure_distance_azimuth(state=pose.as_vector(), features=[feature])
    landmarks = np.zeros((4, 1))

    # Act
    from_slam = inverse_distance_azimuth_slam(
        np.vstack((pose.as_vector(), landmarks, sighting))
    )
    from_map = inverse_distance_azimuth_map(np.vstack((landmarks, sighting)), (pose,))

    # Assert
    np.testing.assert_array_almost_equal(from_slam, [[6.0], [-3.0]])
    np.testing.assert_array_almost_equal(from_map, [[6.0], [-3.0]])


def test_measure_distance_azimuth() -> None:
    """Test the planar distance/azimuth measurement against a known feature."""
    # Arrange
    state = SE3(xyz=np.array([0.0, 0.0, 0.0])).as_vector()
    features = [Feature(id=0, x=3.0, y=4.0)]

    # Act
    measurement = measure_distance_azimuth(state=state, features=features)

    # Assert
    np.testing.assert_array_almost_equal(
        measurement, np.array([[5.0], [np.arctan2(4.0, 3.0)]])
    )


def test_initialize_landmark_estimate_round_trip() -> None:
    """Test that a sighting can be inverted back to the landmark's global position."""
    # Arrange
    pose = SE3(xyz=np.array([1.0, 2.0, 0.0]), roll_pitch_yaw=np.array([0.0, 0.0, 0.5]))
    feature = Feature(id=0, x=6.0, y=-3.0)
    measurement = measure_distance_azimuth(state=pose.as_vector(), features=[feature])
    distance, azimuth = measurement[0, 0], measurement[1, 0]

    # Act
    x, y = initialize_landmark_estimate(pose, distance, azimuth)

    # Assert
    assert x == pytest.approx(feature.x)
    assert y == pytest.approx(feature.y)


def test_measure_distance_azimuth_map_matches_true_geometry() -> None:
    """Test the mapping measurement model against a hand-computed value."""
    # Arrange
    pose = SE3(xyz=np.array([1.0, 1.0, 0.0]))
    landmarks = np.array(
        [[4.0], [5.0], [10.0], [1.0]]
    )  # two landmarks: (4,5) and (10,1)

    # Act
    measurement = measure_distance_azimuth_map(state=landmarks, args=(pose, [0, 1], 2))

    # Assert
    expected_distance = np.array([np.hypot(3.0, 4.0), np.hypot(9.0, 0.0)])
    expected_azimuth = np.array([np.arctan2(4.0, 3.0), np.arctan2(0.0, 9.0)])
    np.testing.assert_array_almost_equal(measurement[0::2, 0], expected_distance)
    np.testing.assert_array_almost_equal(measurement[1::2, 0], expected_azimuth)


def test_step_dynamics_map_is_static() -> None:
    """Test that the mapping motion model leaves landmark positions unchanged."""
    # Arrange
    landmarks = np.array([[4.0], [5.0], [10.0], [1.0]])
    control = np.array([[0.0]])
    state_control = np.vstack((landmarks, control))

    # Act
    result = step_dynamics_map(state_control)

    # Assert
    np.testing.assert_array_almost_equal(result, landmarks)


def test_measure_distance_azimuth_slam_matches_mapping_geometry() -> None:
    """Test that the SLAM measurement model agrees with the mapping model's geometry."""
    # Arrange
    pose = SE3(xyz=np.array([1.0, 1.0, 0.0]))
    landmarks = np.array([[4.0], [5.0], [10.0], [1.0]])
    slam_state = np.vstack((pose.as_vector(), landmarks))

    # Act
    slam_measurement = measure_distance_azimuth_slam(state=slam_state, args=([0, 1], 2))
    map_measurement = measure_distance_azimuth_map(
        state=landmarks, args=(pose, [0, 1], 2)
    )

    # Assert
    np.testing.assert_array_almost_equal(slam_measurement, map_measurement)


def test_step_dynamics_slam_moves_pose_and_freezes_landmarks() -> None:
    """Test that the SLAM motion model steps the pose but not the landmarks."""
    # Arrange
    pose = SE3(xyz=np.array([0.0, 0.0, 0.0]))
    landmarks = np.array([[4.0], [5.0]])
    control = np.array([[1.0], [0.0]])
    state_control = np.vstack((pose.as_vector(), landmarks, control))

    # Act
    result = step_dynamics_slam(state_control, dt=1.0)

    # Assert
    np.testing.assert_array_almost_equal(
        result[0:6], np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    )
    np.testing.assert_array_almost_equal(result[6:], landmarks)
