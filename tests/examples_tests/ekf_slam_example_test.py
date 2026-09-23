"""Add a doc string to my files."""

import numpy as np

from examples.ekf_slam_example import pipeline

POSITION_ERROR_TOLERANCE = 1.0
LANDMARK_MEAN_ERROR_TOLERANCE = 0.5
LANDMARK_MAX_ERROR_TOLERANCE = 1.0


def test_slam_accuracy_on_synthetic_map_and_trajectory() -> None:
    """Test that full SLAM recovers both the pose and the map from sensing alone.

    Neither the pose nor the map is known ahead of time here - both are estimated
    purely from noisy range-azimuth sightings of landmarks along a deterministic,
    seeded trajectory that loops the map enough times for loop closure to kick in.
    """
    # Arrange
    np.random.seed(0)

    # Act
    ekf, true_map, true_pose = pipeline(show_plot=False)

    # Assert
    position_error = np.hypot(ekf.x[0, 0] - true_pose.x, ekf.x[1, 0] - true_pose.y)
    assert position_error < POSITION_ERROR_TOLERANCE

    landmark_errors = [
        np.hypot(
            ekf.x[6 + 2 * feature.id, 0] - feature.x,
            ekf.x[6 + 2 * feature.id + 1, 0] - feature.y,
        )
        for feature in true_map.features
    ]
    assert np.mean(landmark_errors) < LANDMARK_MEAN_ERROR_TOLERANCE
    assert np.max(landmark_errors) < LANDMARK_MAX_ERROR_TOLERANCE
