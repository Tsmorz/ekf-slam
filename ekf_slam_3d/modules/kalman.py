"""Basic docstring for my module."""

import numpy as np

from config.definitions import MEASUREMENT_NOISE, PROCESS_NOISE
from ekf_slam_3d.modules.math_utils import symmetrize_matrix
from ekf_slam_3d.modules.state_space import StateSpaceLinear


class KalmanFilter:
    """Kalman filter implementation."""

    def __init__(
        self,
        state_space: StateSpaceLinear,
        initial_x: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise: np.ndarray | None = None,
        measurement_noise: np.ndarray | None = None,
    ) -> None:
        """Initialize the Kalman Filter.

        :param state_space: linear state space model
        :param initial_x: Initial state estimate
        :param initial_covariance: Initial error covariance
        :param process_noise: Process noise covariance
        :param measurement_noise: Measurement noise covariance
        :return: None
        """
        self.state_space = state_space
        if process_noise is None:
            process_noise = PROCESS_NOISE * np.eye(len(state_space.A))
        self.Q: np.ndarray = process_noise

        if measurement_noise is None:
            measurement_noise = MEASUREMENT_NOISE * np.eye(len(state_space.C))
        self.R: np.ndarray = measurement_noise
        self.x: np.ndarray = initial_x
        self.cov: np.ndarray = initial_covariance

    def predict(self, u: np.ndarray | None = None) -> None:
        """Predict the next state and error covariance.

        :param u: Control input
        """
        if u is None:
            u = np.zeros((self.state_space.B.shape[1], 1))

        self.x = self.state_space.step(x=self.x, u=u)
        self.cov = self.state_space.A @ self.cov @ self.state_space.A.T + self.Q
        self.cov = symmetrize_matrix(self.cov)

    def update(self, z: np.ndarray) -> None:
        """Update the state estimate with measurement z.

        :param z: Measurement
        :return: Updated state estimate and state covariance
        """
        y = z - self.state_space.C @ self.x
        S = self.state_space.C @ self.cov @ self.state_space.C.T + self.R
        K = self.cov @ self.state_space.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        cov = (np.eye(self.cov.shape[0]) - K @ self.state_space.C) @ self.cov
        self.cov = symmetrize_matrix(cov)
