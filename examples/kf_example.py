"""Basic docstring for my module."""

import numpy as np
from loguru import logger

from config.definitions import (
    DEFAULT_DISCRETIZATION,
)
from ekf_slam_3d.data_classes.state_history import StateHistory, plot_history
from ekf_slam_3d.modules.controller import full_state_feedback, get_control_input
from ekf_slam_3d.modules.kalman import KalmanFilter
from ekf_slam_3d.modules.simulators import (
    mass_spring_damper_model,
)
from ekf_slam_3d.modules.state_space import StateSpaceLinear


class KalmanSimulator:
    """Kalman filter implementation."""

    def __init__(
        self,
        state_space: StateSpaceLinear,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
        initial_state: np.ndarray,
    ) -> None:
        """Initialize the Kalman Filter.

        :param state_space: linear state space model
        :param process_noise: Process noise covariance
        :param measurement_noise: Measurement noise covariance
        :param initial_state: Initial state estimate
        :return: None
        """
        self.state_space = state_space
        self.A: np.ndarray = state_space.A
        self.B: np.ndarray = state_space.B
        self.C: np.ndarray = state_space.C
        self.Q: np.ndarray = process_noise
        self.R: np.ndarray = measurement_noise
        self.x: np.ndarray = initial_state

    def step(self, u: np.ndarray | None = None) -> np.ndarray:
        """Predict the next state and error covariance.

        :param u: Control input
        """
        if u is None:
            u = np.zeros((self.B.shape[1], 1))
        scale = np.diag(self.Q)
        scale = np.reshape(scale, (self.Q.shape[0], 1))
        noise = np.random.normal(loc=0.0, scale=scale, size=(self.A.shape[0], 1))
        self.x = self.state_space.step(x=self.x, u=u) + noise
        return self.x

    def get_measurement(self) -> np.ndarray:
        """Get a measurement of the state.

        :return: Measurement of the state
        """
        scale = np.diag(self.R)
        scale = np.reshape(scale, (self.R.shape[0], 1))
        noise = np.random.normal(loc=0.0, scale=scale, size=(self.C.shape[0], 1))
        return self.C @ self.x + noise


def pipeline() -> None:
    """Pipeline to run the repo code."""
    logger.info("Running Kalman Filter pipeline...")

    dt = DEFAULT_DISCRETIZATION
    time = np.arange(0, 10, dt).tolist()
    ss = mass_spring_damper_model(discretization_dt=dt)

    # find the desired control gains
    desired_eigenvalues = np.array([0.89 + 0.29j, 0.89 - 0.29j])
    gain_matrix = full_state_feedback(ss, desired_eigenvalues)
    desired_state = np.array([[0], [0]])

    # initialize the kalman filter
    kf = KalmanFilter(
        state_space=ss,
        process_noise=0.1 * np.eye(len(ss.A)),
        measurement_noise=1.0 * np.eye(len(ss.C)),
        initial_x=np.array([[5.0], [5.0]]),
        initial_covariance=10 * np.eye(len(ss.A)),
    )

    sim = KalmanSimulator(
        state_space=ss,
        process_noise=kf.Q,
        measurement_noise=kf.R,
        initial_state=kf.x,
    )

    sim_history = StateHistory()

    # Generate control inputs, measurements, and update the Kalman filter
    for t in time:
        u = get_control_input(x=kf.x, desired=desired_state, gain_matrix=gain_matrix)

        # Store the updated state for plotting
        sim_history.append_step(t=t, x=kf.x, cov=kf.cov, u=u, x_truth=sim.x)

        # Simulate the system
        sim.step(u=u)
        measurement = sim.get_measurement()

        # Step through the filter
        kf.predict(u=u)
        kf.update(z=measurement)

    plot_history(history=sim_history)


if __name__ == "__main__":
    pipeline()
