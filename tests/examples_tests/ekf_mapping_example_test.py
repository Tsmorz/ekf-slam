"""Add a doc string to my files."""

import numpy as np

from examples.ekf_mapping_example import pipeline

LANDMARK_MEAN_ERROR_TOLERANCE = 0.05
LANDMARK_MAX_ERROR_TOLERANCE = 0.1


def test_mapping_accuracy_on_synthetic_map() -> None:
    """Test that the EKF recovers landmark positions close to their true locations.

    With the robot's pose assumed known (localization solved), the mapping EKF
    should triangulate each landmark's (x, y) position accurately from a handful
    of noisy range-azimuth sightings taken along a deterministic, seeded trajectory.
    """
    # Arrange
    np.random.seed(0)

    # Act
    ekf, true_map, _true_pose = pipeline(show_plot=False)

    # Assert
    errors = [
        np.hypot(
            ekf.x[2 * feature.id, 0] - feature.x,
            ekf.x[2 * feature.id + 1, 0] - feature.y,
        )
        for feature in true_map.features
    ]
    assert np.mean(errors) < LANDMARK_MEAN_ERROR_TOLERANCE
    assert np.max(errors) < LANDMARK_MAX_ERROR_TOLERANCE
