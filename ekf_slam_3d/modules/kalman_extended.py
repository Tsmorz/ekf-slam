"""Basic docstring for my module."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from config.definitions import DEFAULT_CONTROL
from ekf_slam_3d.data_classes.sensors import default_angle_mask
from ekf_slam_3d.modules.math_utils import symmetrize_matrix, wrap_to_pi
from ekf_slam_3d.modules.state_space import StateSpaceLinear, StateSpaceNonlinear


@dataclass(frozen=True, eq=False)
class MeasurementSpec:
    """Optional details of a measurement `z` passed to `ExtendedKalmanFilter.update()`.

    :param covariance: covariance of `z`; defaults to `measurement_noise * I`
    :param angle_mask: boolean mask, True at angle-valued entries of `z` (e.g. azimuth,
        elevation). Those innovation entries are wrapped to [-pi, pi) so a
        measurement/prediction pair straddling the atan2 branch cut isn't treated as a
        huge, spurious correction. Defaults to the sensor's entry in
        `sensors.ANGLE_LAYOUTS`, so it only needs passing for unregistered sensors.
    """

    covariance: np.ndarray | None = None
    angle_mask: np.ndarray | None = None


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
        :param measurement_noise: measurement noise variance (R = measurement_noise * I
            unless `update()` is given an explicit covariance)
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
        spec: MeasurementSpec | None = None,
    ) -> None:
        """Update the state estimate with measurement z.

        :param z: Measurement
        :param sensor: Measurement function
        :param u: Control input
        :param measurement_args: Additional arguments passed through to `sensor`
            (e.g., a list of map features, or a (pose, feature_ids, ...) tuple)
        :param spec: optional covariance of `z` and mask of its angle-valued entries
        :return: Updated state estimate and state covariance
        """
        spec = spec or MeasurementSpec()
        C, _ = self.state_space_nonlinear.linearize(
            model=sensor,
            x=self.x,
            u=u,
            other_args=measurement_args,
        )

        predict_z = (
            sensor(self.x)
            if measurement_args is None
            else sensor(self.x, measurement_args)
        )

        innovation = z - predict_z
        mask = spec.angle_mask
        if mask is None:
            mask = default_angle_mask(sensor, len(z))
        if mask is not None:
            innovation[mask] = wrap_to_pi(innovation[mask])

        R = self._measurement_covariance(z, spec.covariance)
        S = C @ self.cov @ C.T + R
        K = self.cov @ C.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        cov = (np.eye(self.cov.shape[0]) - K @ C) @ self.cov
        self.cov = symmetrize_matrix(cov)

    def initialize_from_measurement(
        self,
        idx: int,
        z: np.ndarray,
        inverse_model: Callable,
        model_args: Any | None = None,
        measurement_covariance: np.ndarray | None = None,
    ) -> None:
        """Seed a not-yet-observed block of the state from a single measurement.

        The block's covariance comes from propagating both the current state
        uncertainty and the measurement noise through the inverse observation model,
        and it keeps the block's cross-covariance with the rest of the state. Without
        that cross-covariance a landmark seeded from an erroneous pose looks like an
        independent, well-known reference, and later observations bend the pose to fit
        it rather than correcting the landmark - leaving the whole map rigidly offset.

        :param idx: index of the first state row of the block being initialized
        :param z: the measurement the block is inverted from
        :param inverse_model: maps stacked [state; z] (and `model_args`) to the block value
        :param model_args: additional arguments passed through to `inverse_model`
        :param measurement_covariance: optional covariance of `z`; defaults to
            `measurement_noise * I`
        """
        G_x, G_z = self.state_space_nonlinear.linearize(
            model=inverse_model, x=self.x, u=z, other_args=model_args
        )
        xz = np.vstack((self.x, z))
        value = (
            inverse_model(xz) if model_args is None else inverse_model(xz, model_args)
        )
        block = slice(idx, idx + value.shape[0])

        R = self._measurement_covariance(z, measurement_covariance)
        cross = G_x @ self.cov
        block_cov = cross @ G_x.T + G_z @ R @ G_z.T

        self.x[block] = value
        self.cov[block, :] = cross
        self.cov[:, block] = cross.T
        self.cov[block, block] = block_cov
        self.cov = symmetrize_matrix(self.cov)

    def _measurement_covariance(
        self, z: np.ndarray, measurement_covariance: np.ndarray | None
    ) -> np.ndarray:
        if measurement_covariance is not None:
            return measurement_covariance
        return self.measurement_noise * np.eye(len(z))
