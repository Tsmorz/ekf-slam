"""Basic docstring for my module."""

from collections.abc import Callable
from typing import Any

import numpy as np

from config.definitions import DEFAULT_CONTROL
from ekf_slam_3d.modules.math_utils import symmetrize_matrix
from ekf_slam_3d.modules.state_space import StateSpaceLinear, StateSpaceNonlinear


class ExtendedKalmanFilter:
    """Extended Kalman filter implementation."""

    def __init__(
        self,
        state_space_nonlinear: StateSpaceNonlinear,
        initial_x: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise: np.ndarray,
        measurement_noise: float,
    ) -> None:
        """Initialize the Extended Kalman Filter.

        :param state_space_nonlinear: nonlinear state space model
        :param initial_x: Initial state estimate
        :param initial_covariance: Initial error covariance
        :param process_noise: Process noise covariance
        :return: None
        """
        self.state_space_nonlinear = state_space_nonlinear
        self.x: np.ndarray = initial_x
        self.cov: np.ndarray = initial_covariance
        self.Q: np.ndarray = process_noise
        self.measurement_noise = measurement_noise

    def predict(self, u: np.ndarray = DEFAULT_CONTROL) -> None:
        """Predict the next state and error covariance.

        :param u: Control input
        """
        A, B = self.state_space_nonlinear.linearize(
            model=self.state_space_nonlinear.motion_model, x=self.x, u=u
        )
        ss = StateSpaceLinear(A, B)

        self.x = self.state_space_nonlinear.step(x=self.x, u=u)
        self.cov = ss.A @ self.cov @ ss.A.T + ss.B @ self.Q @ ss.B.T
        self.cov = symmetrize_matrix(self.cov)

    def update(
        self,
        z: np.ndarray,
        sensor: Callable,
        u: np.ndarray,
        measurement_args: Any | None = None,
    ) -> None:
        """Update the state estimate with measurement z.

        :param z: Measurement
        :param sensor: Measurement function
        :param u: Control input
        :param measurement_args: Additional arguments passed through to `sensor`
            (e.g., a list of map features, or a (pose, feature_ids, ...) tuple)
        :return: Updated state estimate and state covariance
        """
        A, B = self.state_space_nonlinear.linearize(
            model=self.state_space_nonlinear.motion_model,
            x=self.x,
            u=u,
        )
        C, D = self.state_space_nonlinear.linearize(
            model=sensor,
            x=self.x,
            u=u,
            other_args=measurement_args,
        )
        state_space = StateSpaceLinear(A, B, C, D)

        predict_z = (
            sensor(self.x)
            if measurement_args is None
            else sensor(self.x, measurement_args)
        )

        R = self.measurement_noise * np.eye(len(z))
        S = state_space.C @ self.cov @ state_space.C.T + R
        K = self.cov @ state_space.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - predict_z)
        cov = (np.eye(self.cov.shape[0]) - K @ state_space.C) @ self.cov
        self.cov = symmetrize_matrix(cov)
