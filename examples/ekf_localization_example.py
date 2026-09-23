"""Basic docstring for my module."""

import numpy as np
from loguru import logger

from config.definitions import (
    CONTROL_NOISE_COVARIANCE,
    LOG_DECIMALS,
    MEASUREMENT_NOISE,
)
from ekf_slam_3d.data_classes.lie_algebra import SE3, state_to_se3
from ekf_slam_3d.data_classes.map import make_random_map_planar
from ekf_slam_3d.data_classes.sensors import Sensor, angle_mask, step_dynamics
from ekf_slam_3d.modules.kalman_extended import ExtendedKalmanFilter, MeasurementSpec
from ekf_slam_3d.modules.simulators import SlamSimulator
from ekf_slam_3d.modules.state_space import StateSpaceNonlinear


def pipeline(
    show_plot: bool, num_steps: int | None = None
) -> tuple[ExtendedKalmanFilter, SE3]:
    """Run the EKF for localization: the map is known, only the pose is estimated.

    :param show_plot: whether to render the live simulation
    :param num_steps: number of simulation steps to run (defaults to the full demo run)
    :return: (ekf holding the final pose estimate, the final true pose)
    """
    logger.info("Running EKF pipeline for localization.")
    robot_pose = SE3(xyz=np.zeros((3,)), roll_pitch_yaw=np.zeros((3,)))
    robot_pose.x = 20
    meas = np.array([])

    ekf = ExtendedKalmanFilter(
        state_space_nonlinear=StateSpaceNonlinear(motion_model=step_dynamics),
        initial_x=robot_pose.as_vector(),
        initial_covariance=0.1 * np.eye(robot_pose.as_vector().shape[0]),
        process_noise=CONTROL_NOISE_COVARIANCE,
        measurement_noise=MEASUREMENT_NOISE,
    )

    sim = SlamSimulator(
        state_space_nl=ekf.state_space_nonlinear,
        process_noise=ekf.Q,
        initial_pose=robot_pose,
        sim_map=make_random_map_planar(num_features=10, dim=(40, 40)),
    )

    time_stamps = sim.time_stamps if num_steps is None else sim.time_stamps[:num_steps]
    for time in time_stamps:
        control_input = np.array([[1.0], [2 * np.pi / 100]])
        sim.step(u=control_input)
        ekf.predict(u=control_input)

        # update the state estimate with the measurements
        _frac, whole = np.modf(time)
        if whole % 3 == 0 and time > 0.0:
            # TODO - update the heading with the magnetometer
            ekf.x[5, 0] = sim.pose.yaw + np.random.normal(0, 0.1)

        if whole % 15 == 0 and time > 0.0:
            meas = Sensor.DIST_AZI_ELE.func(
                state=sim.pose.as_vector(), features=sim.map.features
            )
            ekf.update(
                z=meas,
                sensor=Sensor.DIST_AZI_ELE.func,
                u=control_input,
                measurement_args=sim.map.features,
                spec=MeasurementSpec(
                    angle_mask=angle_mask(len(meas), stride=3, offsets=(1, 2))
                ),
            )

        sim.append_result(
            estimate=(state_to_se3(state=ekf.x), ekf.cov),
            measurement=meas,
            show_plot=show_plot,
        )

        if show_plot:
            x_str = np.array2string(ekf.x.T, precision=LOG_DECIMALS, floatmode="fixed")
            logger.info(f"state: {x_str}")

    return ekf, sim.pose


if __name__ == "__main__":
    pipeline(show_plot=True)
