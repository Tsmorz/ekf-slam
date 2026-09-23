"""Basic docstring for my module."""

import numpy as np
from loguru import logger
from scipy.signal import place_poles

from config.definitions import (
    DEFAULT_NUM_STEPS,
    DELTA_T,
    MAX_PITCH_RATE,
    MAX_TURN_RATE,
    PURE_PURSUIT_LOOKAHEAD_STEPS,
    ROBOT_SPEED,
)
from ekf_slam_3d.data_classes.lie_algebra import SE3
from ekf_slam_3d.data_classes.sensors import step_dynamics
from ekf_slam_3d.modules.math_utils import wrap_to_pi
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


def box_path(
    start: SE3,
    side_steps: int,
    radius_steps: int,
    turn: int = 1,
    speed: float = ROBOT_SPEED,
) -> np.ndarray:
    """Return the positions of one noise-free loop of a rounded box path.

    :param start: pose the loop starts (and ends) at
    :param side_steps: straight steps per side
    :param radius_steps: steps per rounded 90 degree corner
    :param turn: +1 for counterclockwise loops, -1 for clockwise
    :param speed: forward speed
    :return: (N, 2) array of path positions, N = 4 * (side_steps + radius_steps)
    """
    corner_rate = turn * np.pi / 2 / (radius_steps * DELTA_T)
    one_side = side_steps * [0.0] + radius_steps * [corner_rate]
    state = start.as_vector()
    points = [[start.x, start.y]]
    for turn_rate in 4 * one_side:
        state = step_dynamics(np.vstack((state, [[speed], [turn_rate]])))
        points.append([state[0, 0], state[1, 0]])
    return np.array(points[:-1])


def pure_pursuit_turn_rate(
    pose: SE3,
    path: np.ndarray,
    lookahead_steps: int = PURE_PURSUIT_LOOKAHEAD_STEPS,
    speed: float = ROBOT_SPEED,
    max_turn_rate: float = MAX_TURN_RATE,
) -> float:
    """Return the turn rate that steers `pose` along a closed path (pure pursuit).

    :param pose: the pose to steer from (in practice the filter's estimate)
    :param path: (N, 2) closed-loop path positions
    :param lookahead_steps: how many path points past the nearest one to aim at
    :param speed: forward speed
    :param max_turn_rate: turn rate limit
    :return: turn rate command (rad per unit time)
    """
    target = _lookahead_target(pose, path, lookahead_steps)
    dx, dy = target[0] - pose.x, target[1] - pose.y
    alpha = wrap_to_pi(np.arctan2(dy, dx) - pose.yaw)
    turn_rate = 2 * speed * np.sin(alpha) / max(float(np.hypot(dx, dy)), 1e-6)
    return float(np.clip(turn_rate, -max_turn_rate, max_turn_rate))


def pitch_rate_toward_path(
    pose: SE3,
    path: np.ndarray,
    lookahead_steps: int = PURE_PURSUIT_LOOKAHEAD_STEPS,
    max_pitch_rate: float = MAX_PITCH_RATE,
) -> float:
    """Return the pitch rate that points `pose` at the altitude of the lookahead point.

    :param pose: the pose to steer from (in practice the filter's estimate)
    :param path: (N, 3) closed-loop path positions, with altitude in the last column
    :param lookahead_steps: how many path points past the nearest one to aim at
    :param max_pitch_rate: pitch rate limit
    :return: pitch rate command (rad per unit time)
    """
    target = _lookahead_target(pose, path, lookahead_steps)
    horizontal = max(float(np.hypot(target[0] - pose.x, target[1] - pose.y)), 1e-6)
    desired_pitch = np.arctan2(target[2] - pose.z, horizontal)
    pitch_rate = (desired_pitch - pose.pitch) / DELTA_T
    return float(np.clip(pitch_rate, -max_pitch_rate, max_pitch_rate))


def _lookahead_target(pose: SE3, path: np.ndarray, lookahead_steps: int) -> np.ndarray:
    nearest = int(np.argmin(np.hypot(path[:, 0] - pose.x, path[:, 1] - pose.y)))
    return path[(nearest + lookahead_steps) % len(path)]
