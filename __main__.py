"""Basic docstring for my module."""

import argparse
from enum import Enum, auto

import numpy as np
from loguru import logger

from config.definitions import DEFAULT_DISCRETIZATION, MEASUREMENT_NOISE, PROCESS_NOISE
from src.data_classes.state_space_data import plot_history, plot_states
from src.modules.controller import full_state_feedback, get_control_input
from src.modules.kalman import KalmanFilter
from src.modules.simulator import Simulator
from src.modules.state_space import StateSpaceData, mass_spring_damper_discrete


class Pipeline(Enum):
    """Create an enumerator to choose which pipeline to run."""

    KF = auto()
    EKF = auto()
    EKF_SLAM = auto()
    CONTROLLER = auto()


def run_kf_pipeline() -> None:
    """Pipeline to run the repo code."""
    logger.info("Running Kalman Filter pipeline...")

    dt = DEFAULT_DISCRETIZATION
    time = np.arange(0, 10, dt).tolist()
    ss = mass_spring_damper_discrete(discretization_dt=dt)

    # find the desired control gains
    desired_eigenvalues = np.array([0.89 + 0.29j, 0.89 - 0.29j])
    gain_matrix = full_state_feedback(ss, desired_eigenvalues)
    desired_state = np.array([[0], [0]])

    # initialize the kalman filter
    kf = KalmanFilter(
        state_space=ss,
        process_noise=PROCESS_NOISE * np.eye(2),
        measurement_noise=MEASUREMENT_NOISE * np.eye(2),
        initial_state=np.array([[5.0], [5.0]]),
        initial_covariance=5 * np.eye(2),
    )

    sim = Simulator(
        state_space=ss,
        process_noise=kf.Q,
        measurement_noise=kf.R,
        initial_state=kf.x,
    )

    sim_history = StateSpaceData()

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
    plot_states(history=sim_history)


def main(pipeline_num: int) -> None:
    """Process which pipeline to run."""
    if pipeline_num == Pipeline.KF.value:
        run_kf_pipeline()
    else:
        logger.error(f"Invalid pipeline number: {pipeline_num}")


if __name__ == "__main__":  # pragma: no cover
    """Run the main program with this function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pipeline",
        action="store",
        type=int,
        required=True,
        help="Choose which pipeline to run. (1, 2, 3, etc.)",
    )
    args = parser.parse_args()
    main(pipeline_num=args.pipeline)
