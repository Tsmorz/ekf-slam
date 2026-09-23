"""Add a doc string to my files."""

import numpy as np

from examples.ekf_localization_example import pipeline

POSITION_ERROR_TOLERANCE = 1.0


def test_localization_accuracy_on_synthetic_trajectory() -> None:
    """Test that the EKF localizes a simulated robot close to its true position.

    This drives a deterministic, seeded synthetic trajectory and noisy sensor
    readings through the localization pipeline, then checks the final position
    estimate against the known ground truth.
    """
    # Arrange
    np.random.seed(0)

    # Act
    ekf, true_pose = pipeline(show_plot=False, num_steps=300)

    # Assert
    position_error = np.hypot(ekf.x[0, 0] - true_pose.x, ekf.x[1, 0] - true_pose.y)
    assert position_error < POSITION_ERROR_TOLERANCE
