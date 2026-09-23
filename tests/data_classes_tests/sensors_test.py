"""Add a doc string to my files."""

import numpy as np
import pytest

from ekf_slam_3d.data_classes.lie_algebra import SE3
from ekf_slam_3d.data_classes.map import Feature
from ekf_slam_3d.data_classes.sensors import (
    Sensor,
    angle_mask,
    default_angle_mask,
    distance_azimuth_covariance,
    features_in_range,
    initialize_landmark_estimate,
    initialize_landmark_estimate_3d,
    inverse_distance_azimuth_elevation_slam,
    inverse_distance_azimuth_map,
    inverse_distance_azimuth_slam,
    measure_distance_azimuth,
    measure_distance_azimuth_elevation,
    measure_distance_azimuth_elevation_slam,
    measure_distance_azimuth_map,
    measure_distance_azimuth_slam,
    measure_elevation,
    step_dynamics,
    step_dynamics_3d,
    step_dynamics_map,
    step_dynamics_slam,
    step_dynamics_slam_3d,
)


def test_measure_elevation_is_angle_above_the_horizontal() -> None:
    """Test that elevation ignores heading and doesn't flip when a feature is behind."""
    # Arrange
    pose = SE3(roll_pitch_yaw=np.array([0.0, 0.0, 2.0]))
    features = [
        Feature(id=0, x=3.0, y=4.0, z=5.0),
        Feature(id=1, x=-3.0, y=-4.0, z=-5.0),
    ]

    # Act
    elevation = measure_elevation(pose.as_vector(), features)

    # Assert
    np.testing.assert_array_almost_equal(elevation[:, 0], [np.pi / 4, -np.pi / 4])


def test_3d_sighting_round_trip() -> None:
    """Test that a range-azimuth-elevation sighting inverts back to the landmark."""
    # Arrange
    pose = SE3(xyz=np.array([1.0, 2.0, 3.0]), roll_pitch_yaw=np.array([0.0, 0.1, 0.5]))
    feature = Feature(id=0, x=6.0, y=-3.0, z=7.5)
    sighting = measure_distance_azimuth_elevation(pose.as_vector(), [feature])
    landmarks = np.zeros((6, 1))

    # Act
    xyz = initialize_landmark_estimate_3d(pose, *sighting[:, 0])
    from_slam = inverse_distance_azimuth_elevation_slam(
        np.vstack((pose.as_vector(), landmarks, sighting))
    )

    # Assert
    np.testing.assert_array_almost_equal(xyz, [6.0, -3.0, 7.5])
    np.testing.assert_array_almost_equal(from_slam, [[6.0], [-3.0], [7.5]])


def test_3d_slam_measurement_matches_the_known_map_sensor() -> None:
    """Test that the 3D SLAM model reads landmarks out of the state consistently."""
    # Arrange
    pose = SE3(xyz=np.array([1.0, 1.0, 2.0]), roll_pitch_yaw=np.array([0.0, 0.0, 0.3]))
    features = [Feature(id=0, x=4.0, y=5.0, z=6.0), Feature(id=1, x=10.0, y=1.0, z=0.0)]
    landmarks = np.array([[f.x, f.y, f.z] for f in features]).reshape(-1, 1)

    # Act
    from_state = measure_distance_azimuth_elevation_slam(
        np.vstack((pose.as_vector(), landmarks)), args=([1, 0], 2)
    )
    from_map = measure_distance_azimuth_elevation(pose.as_vector(), features[::-1])

    # Assert
    np.testing.assert_array_almost_equal(from_state, from_map)


def test_step_dynamics_3d_climbs_with_pitch() -> None:
    """Test that the 3D motion model climbs along its pitch and applies pitch rate."""
    # Arrange
    pose = SE3(roll_pitch_yaw=np.array([0.0, np.pi / 6, 0.0]))
    controls = np.array([[2.0], [0.1], [0.05]])

    # Act
    result = step_dynamics_3d(np.vstack((pose.as_vector(), controls)), dt=1.0)

    # Assert
    np.testing.assert_array_almost_equal(
        result[:, 0], [2.0 * np.cos(np.pi / 6), 0.0, 1.0, 0.0, np.pi / 6 + 0.05, 0.1]
    )


def test_planar_dynamics_is_3d_dynamics_without_pitch_rate() -> None:
    """Test that the 2-control model matches the 3D one with zero pitch rate."""
    # Arrange
    pose = SE3(xyz=np.array([1.0, 2.0, 3.0]), roll_pitch_yaw=np.array([0.0, 0.2, 0.7]))
    controls = np.array([[1.5], [0.3]])

    # Act
    planar = step_dynamics(np.vstack((pose.as_vector(), controls)))
    spatial = step_dynamics_3d(np.vstack((pose.as_vector(), controls, [[0.0]])))

    # Assert
    np.testing.assert_array_almost_equal(planar, spatial)


def test_step_dynamics_slam_3d_moves_pose_and_freezes_landmarks() -> None:
    """Test that the 3D SLAM motion model steps the pose but not the landmarks."""
    # Arrange
    landmarks = np.array([[4.0], [5.0], [6.0]])
    controls = np.array([[1.0], [0.0], [0.0]])

    # Act
    result = step_dynamics_slam_3d(np.vstack((SE3().as_vector(), landmarks, controls)))

    # Assert
    np.testing.assert_array_almost_equal(result[0:3, 0], [1.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(result[6:], landmarks)


def test_angle_mask_marks_interleaved_angles() -> None:
    """Test that the mask flags the azimuth/elevation slots of each group."""
    # Act
    mask = angle_mask(length=6, stride=3, offsets=(1, 2))

    # Assert
    np.testing.assert_array_equal(mask, [False, True, True, False, True, True])


def test_default_angle_mask_follows_each_sensors_layout() -> None:
    """Test that registered sensors get their angle mask and others get none."""
    # Act / Assert
    np.testing.assert_array_equal(
        default_angle_mask(Sensor.DIST_AZI_ELE.func, 6),
        [False, True, True, False, True, True],
    )
    np.testing.assert_array_equal(
        default_angle_mask(measure_distance_azimuth_slam, 4), [False, True, False, True]
    )
    np.testing.assert_array_equal(
        default_angle_mask(measure_distance_azimuth_elevation_slam, 3),
        [False, True, True],
    )
    assert default_angle_mask(Sensor.GPS.func, 3) is None


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


def test_features_in_range_counts_height() -> None:
    """Test that sensing range is a 3D distance."""
    # Arrange
    features = [Feature(id=0, x=3.0, y=4.0, z=12.0)]

    # Act / Assert
    assert features_in_range(SE3(), features, max_range=10.0) == []
    assert features_in_range(SE3(), features, max_range=13.0) == [0]


def test_distance_azimuth_elevation_covariance() -> None:
    """Test that both angles of a [d, az, el] triple get the distance-scaled variance."""
    # Act
    covariance = distance_azimuth_covariance(
        np.array([[4.0], [0.3], [0.1]]), noise=0.1, stride=3
    )

    # Assert
    np.testing.assert_array_almost_equal(
        np.diag(covariance), [0.01, (0.1 / 5) ** 2, (0.1 / 5) ** 2]
    )


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
