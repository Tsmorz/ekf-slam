"""Basic docstring for my module."""

import numpy as np
from loguru import logger
from scipy.signal import place_poles

from config.definitions import DEFAULT_NUM_STEPS
from ekf_slam_3d.modules.state_space import StateSpaceLinear


def full_state_feedback(state_space: StateSpaceLinear, desired_eigenvalues: np.ndarray):
    """Calculate the feedback gains for a desired response.

    :param state_space: State-space model
    :param desired_eigenvalues: Desired eigenvalues
    :return: Feedback gains
    """
    place_result = place_poles(state_space.A, state_space.B, desired_eigenvalues)
    K = place_result.gain_matrix

    augmented = state_space.A - state_space.B @ K
    if np.linalg.eigvals(augmented).all() != desired_eigenvalues.all():
        msg = "The desired eigenvalues are not correct."
        logger.error(msg)
        raise ValueError(msg)
    return K


def get_control_input(
    x: np.ndarray,
    desired: np.ndarray,
    gain_matrix: np.ndarray,
    limit: float = np.inf,
) -> np.ndarray:
    """Calculate the control input based on the state and desired tracking.

    :param x: Current state
    :param desired: Desired state
    :param gain_matrix: Feedback gain matrix
    :param limit: Limit on the controller
    :return: Control input
    """
    control = -gain_matrix @ (x - desired)
    return np.clip(control, -limit, limit)


class LQRController:
    """Discrete-time LQR Controller with finite horizon."""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        num_steps: int = DEFAULT_NUM_STEPS,
    ):
        """Initialize the LQR controller.

        :param A: State matrix
        :param B: Input matrix
        :param Q: State cost matrix
        :param R: Control cost matrix
        :param num_steps: Time horizon
        """
        if A.shape[0] != B.shape[0]:
            msg = "A and B must have the same number of rows"
            logger.error(msg)
            raise ValueError(msg)
        self.A = A
        self.B = B

        if Q is None:
            Q = np.eye(A.shape[0])  # Default Q matrix
        if R is None:
            R = np.eye(B.shape[1])  # Default R matrix

        self.Q = Q
        self.R = R

        self.num_steps = num_steps
        self.K = self.compute_finite_horizon_lqr()

    def compute_finite_horizon_lqr(self):
        """Compute the finite-horizon LQR gains using backward recursion.

        :return: List of state feedback gains for each time step.
        """
        cost = self.Q  # Initialize terminal cost
        gain_list = []

        for _ in range(self.num_steps):
            gain = np.linalg.inv(self.B.T @ cost @ self.B + self.R) @ (
                self.B.T @ cost @ self.A
            )
            gain_list.insert(0, gain)  # Store gain for each time step
            cost = self.Q + self.A.T @ cost @ (self.A - self.B @ gain)

        return gain_list

    def get_control_gain(self, t):
        """Get the LQR gain at time step t.

        :param t: Current time step
        :return: State feedback gain K
        """
        if t >= self.num_steps:
            return self.K[-1]  # Use the last computed gain after horizon
        return self.K[t]


def get_angular_velocities_for_box(steps: int, radius_steps: int) -> list[float]:
    """Create the angular velocity control inputs for a box."""
    side_length = int(steps / 4)
    one_side = side_length * [0] + radius_steps * [np.pi / 2 / radius_steps]
    turning_rates = one_side + one_side + one_side + one_side
    return turning_rates
