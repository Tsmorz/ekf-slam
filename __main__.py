"""Basic docstring for my module."""

import argparse
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from config.definitions import (
    DEFAULT_DISCRETIZATION,
)
from src.data_classes.lie_algebra import SE3
from src.data_classes.map import make_random_map_planar
from src.data_classes.sensors import SensorType
from src.data_classes.state_history import StateHistory, plot_history
from src.modules.controller import full_state_feedback, get_control_input
from src.modules.extended_kalman import ExtendedKalmanFilter
from src.modules.kalman import KalmanFilter
from src.modules.simulator import (
    KalmanSimulator,
    SlamSimulator,
    add_measurement_to_plot,
    mass_spring_damper_model,
    robot_model,
)


class Pipeline(Enum):
    """Create an enumerator to choose which pipeline to run."""

    KF = auto()
    EKF = auto()
    EKF_SLAM = auto()
    CONTROLLER = auto()
    STATE_SPACE = auto()


def run_state_space_pipeline() -> None:
    """Run the main program with this function."""
    dt = DEFAULT_DISCRETIZATION
    ss_model = mass_spring_damper_model(discretization_dt=dt)
    ss_model.step_response(delta_t=dt, plot_response=True)
    ss_model.impulse_response(delta_t=dt, plot_response=True)


def run_kf_pipeline() -> None:
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
        initial_x=np.array([[5.0], [5.0]]),
        initial_covariance=5 * np.eye(2),
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


def run_ekf_pipeline() -> None:
    """Test the EKF algorithm."""

    def state_to_se2(state: np.ndarray) -> SE3:
        """Map the state vector to SE2."""
        return SE3(xyz=state[0:3, 0], roll_pitch_yaw=state[3:6, 0])

    logger.info("Running Extended Kalman Filter pipeline.")

    robot = robot_model()
    initial_pose = SE3(xyz=np.zeros(3), roll_pitch_yaw=np.zeros(3))
    ekf = ExtendedKalmanFilter(
        state_space_nonlinear=robot,
        initial_x=initial_pose.as_vector(),
        initial_covariance=3 * np.eye(6),
    )

    sim = SlamSimulator(
        state_space_nl=robot,
        process_noise=ekf.Q,
        measurement_noise=ekf.R,
        initial_pose=initial_pose,
        steps=100,
    )

    sim_map = make_random_map_planar(num_features=3, dim=(15, 15))

    for time, omega in enumerate(sim.controls):
        u = np.array([[0.5], [omega]])
        sim.step(u=u)
        ekf.predict(u=u)

        if (time / 20) % 1 == 0 and time != 0:
            for feature in sim_map.features:
                measurement = sim.get_measurement(
                    feature=feature,
                    sensor_type=SensorType.DISTANCE_AND_BEARING,
                )
                logger.info(f"Measurement={measurement}")

                ekf.update(z=measurement.as_vector(), u=u, measurement_args=feature)

                if measurement.type == SensorType.DISTANCE_AND_BEARING:
                    add_measurement_to_plot(measurement, state=ekf.x)

        pose = state_to_se2(state=ekf.x)
        sim.append_estimate(estimated_pose=pose, plot_pose=True)

    plt.show()
    plt.close()


if __name__ == "__main__":  # pragma: no cover
    """Run the main program with this function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pipeline",
        action="store",
        type=str,
        required=True,
        help=f"Choose which pipeline to run - {[v.name for v in Pipeline]}",
    )
    args = parser.parse_args()

    pipeline_id = args.pipeline

    if pipeline_id == Pipeline.KF.name:
        run_kf_pipeline()
    elif pipeline_id == Pipeline.EKF.name:
        run_ekf_pipeline()
    elif pipeline_id == Pipeline.STATE_SPACE.name:
        run_state_space_pipeline()
    else:
        msg = f"Invalid pipeline number: {pipeline_id}"
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Program complete.")
