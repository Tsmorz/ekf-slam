"""Add a doc string to my files."""

import numpy as np
import pytest

from ekf_slam_3d.data_classes.lie_algebra import SE3
from ekf_slam_3d.modules.controller import (
    box_path,
    full_state_feedback,
    get_control_input,
    pitch_rate_toward_path,
    pure_pursuit_turn_rate,
)
from ekf_slam_3d.modules.simulators import mass_spring_damper_model


@pytest.mark.parametrize("turn", [1, -1])
def test_box_path_is_a_closed_loop(turn: int) -> None:
    """Test that one loop of the box path returns to its start, turning the right way."""
    # Arrange
    start = SE3(xyz=np.array([2.0, 3.0, 0.0]))

    # Act
    path = box_path(start, side_steps=5, radius_steps=4, turn=turn)

    # Assert
    assert path.shape == (4 * (5 + 4), 2)
    np.testing.assert_array_almost_equal(path[0], [2.0, 3.0])
    step_back_to_start = np.hypot(*(path[-1] - path[0]))
    assert step_back_to_start == pytest.approx(1.0, abs=0.05)
    assert np.sign(path[:, 1].mean() - 3.0) == turn


@pytest.mark.parametrize(("target_y", "expected_sign"), [(2.0, 1.0), (-2.0, -1.0)])
def test_pure_pursuit_turns_toward_the_path(
    target_y: float, expected_sign: float
) -> None:
    """Test that pure pursuit steers left for a path to the left, right for one to the right."""
    # Arrange
    path = np.array([[x, target_y] for x in np.arange(0.0, 10.0)])

    # Act
    turn_rate = pure_pursuit_turn_rate(SE3(), path, lookahead_steps=2)

    # Assert
    assert np.sign(turn_rate) == expected_sign


def test_pure_pursuit_goes_straight_when_on_the_path() -> None:
    """Test that pure pursuit doesn't turn when already heading along the path."""
    # Arrange
    path = np.array([[x, 0.0] for x in np.arange(0.0, 10.0)])

    # Act
    turn_rate = pure_pursuit_turn_rate(SE3(), path, lookahead_steps=3)

    # Assert
    assert turn_rate == pytest.approx(0.0)


def test_pure_pursuit_turn_rate_is_limited() -> None:
    """Test that the turn rate is clipped when the target is sharply to the side."""
    # Arrange
    path = np.array([[0.0, 0.5], [0.0, 1.0], [0.0, 1.5]])

    # Act
    turn_rate = pure_pursuit_turn_rate(
        SE3(), path, lookahead_steps=1, max_turn_rate=0.2
    )

    # Assert
    assert abs(turn_rate) == pytest.approx(0.2)


def test_get_control_input() -> None:
    """Test that the control input is calculated correctly."""
    # Arrange
    x = np.array([[2.0], [2.0]])
    desired = np.array([[1.0], [1.0]])
    gain_matrix = np.array([[1.0, 1.0]])
    limit = 10.0

    # Act
    control = get_control_input(x, desired, gain_matrix, limit)

    # Assert
    np.testing.assert_array_almost_equal(control, np.array([-sum(x - desired)]))


def test_full_state_feedback() -> None:
    """Test that the full state feedback is calculated correctly."""
    # Arrange
    state_space = mass_spring_damper_model()
    desired_eigenvalues = np.array([1.0, 2.0])

    # Act
    gain_matrix = full_state_feedback(state_space, desired_eigenvalues)
    A_prime = state_space.A - state_space.B @ gain_matrix

    # Assert
    new_eigenvalues = np.linalg.eigvals(A_prime)
    np.testing.assert_array_almost_equal(new_eigenvalues, desired_eigenvalues)


@pytest.mark.parametrize(("target_z", "expected_sign"), [(3.0, 1.0), (-3.0, -1.0)])
def test_pitch_rate_toward_path_climbs_and_descends(
    target_z: float, expected_sign: float
) -> None:
    """Test that the pitch command noses up toward a higher path and down toward a lower one."""
    # Arrange
    path = np.array([[x, 0.0, target_z] for x in np.arange(0.0, 10.0)])

    # Act
    pitch_rate = pitch_rate_toward_path(SE3(), path, lookahead_steps=3)

    # Assert
    assert np.sign(pitch_rate) == expected_sign


def test_pitch_rate_is_zero_when_level_on_a_level_path() -> None:
    """Test that a level vehicle on a level path isn't commanded to pitch."""
    # Arrange
    path = np.array([[x, 0.0, 0.0] for x in np.arange(0.0, 10.0)])

    # Act / Assert
    assert pitch_rate_toward_path(SE3(), path) == pytest.approx(0.0)
