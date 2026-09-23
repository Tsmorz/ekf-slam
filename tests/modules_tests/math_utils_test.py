"""Add a doc string to my files."""

import numpy as np
import pytest

from ekf_slam_3d.modules.math_utils import matrix_exponential, wrap_to_pi


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, 0.0),
        (np.pi, -np.pi),
        (-np.pi, -np.pi),
        (3 * np.pi, -np.pi),
        (2 * np.pi + 0.1, 0.1),
        (-2 * np.pi - 0.1, -0.1),
    ],
)
def test_wrap_to_pi(angle: float, expected: float) -> None:
    """Test that angles are wrapped into [-pi, pi)."""
    # Act / Assert
    assert wrap_to_pi(np.array([angle]))[0] == pytest.approx(expected)


@pytest.mark.parametrize("t", [1.0, 0.1, 0.01])
def test_matrix_exponential_jordan(t):
    """Test matrix exponential function with different time lengths."""
    # Arrange
    matrix = np.array(
        [
            [0, 1],
            [0, 0],
        ]
    )
    expected = np.array(
        [
            [1, t],
            [0, 1],
        ]
    )

    # Act
    mat_exp = matrix_exponential(matrix, t=t)

    # Assert
    np.testing.assert_array_almost_equal(mat_exp, expected, decimal=3)


@pytest.mark.parametrize("t", [1.0, 0.1, 0.01])
def test_matrix_exponential_diagonal(t: float) -> None:
    """Test matrix exponential function with different time lengths."""
    # Arrange
    matrix = np.eye(5)
    expected = np.exp(t) * matrix

    # Act
    mat_exp = matrix_exponential(matrix, t=t)

    # Assert
    np.testing.assert_array_almost_equal(mat_exp, expected, decimal=3)


def test_matrix_exponential_nonsquare() -> None:
    """Test matrix exponential function with different time lengths."""
    # Arrange
    matrix_nonsquare = np.ones((3, 2))

    # Act / Assert
    with np.testing.assert_raises(ValueError):
        _ = matrix_exponential(matrix_nonsquare)
