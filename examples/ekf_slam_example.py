"""Basic docstring for my module."""

import numpy as np
from loguru import logger

from config.definitions import (
    LANDMARK_INIT_VARIANCE,
    LOG_DECIMALS,
    MEASUREMENT_NOISE,
    SIGMA_OMEGA,
    SIGMA_VEL,
)
from ekf_slam_3d.data_classes.lie_algebra import SE3, state_to_se3
from ekf_slam_3d.data_classes.map import Feature, Map, make_box_map_planar
from ekf_slam_3d.data_classes.sensors import (
    initialize_landmark_estimate,
    measure_distance_azimuth,
    measure_distance_azimuth_slam,
    step_dynamics,
    step_dynamics_slam,
)
from ekf_slam_3d.modules.controller import get_angular_velocities_for_box
from ekf_slam_3d.modules.kalman_extended import ExtendedKalmanFilter
from ekf_slam_3d.modules.simulators import SlamSimulator
from ekf_slam_3d.modules.state_space import StateSpaceNonlinear


def pipeline(
    show_plot: bool, num_loops: int = 5
) -> tuple[ExtendedKalmanFilter, Map, SE3]:
    """Run the EKF for full SLAM: the pose and the map are estimated together.

    Unlike the mapping pipeline, neither the pose nor the map is known ahead of time -
    the EKF state packs the pose (0:6) and interleaved x/y landmark pairs (6:), and
    both are corrected purely from range-azimuth sightings of the landmarks.

    :param show_plot: whether to render the live simulation
    :param num_loops: number of times to drive the box trajectory around the map
    :return: (ekf holding the final pose + map estimate, the true map, the final true pose)
    """
    logger.info("Running EKF pipeline for SLAM.")

    true_map = make_box_map_planar(num_features=8, dim=(20, 20))
    num_features = len(true_map.features)

    initial_pose = SE3(xyz=np.array([10.0, 10.0, 0.0]))
    initial_x = np.vstack((initial_pose.as_vector(), np.zeros((2 * num_features, 1))))
    initial_cov = 0.01 * np.eye(6 + 2 * num_features)
    initial_cov[6:, 6:] = LANDMARK_INIT_VARIANCE * np.eye(2 * num_features)

    process_noise = np.array([[SIGMA_VEL, 0.0], [0.0, SIGMA_OMEGA]])
    ekf = ExtendedKalmanFilter(
        state_space_nonlinear=StateSpaceNonlinear(motion_model=step_dynamics_slam),
        initial_x=initial_x,
        initial_covariance=initial_cov,
        process_noise=process_noise,
        measurement_noise=MEASUREMENT_NOISE,
    )

    sim = SlamSimulator(
        state_space_nl=StateSpaceNonlinear(motion_model=step_dynamics),
        process_noise=process_noise,
        initial_pose=initial_pose,
        sim_map=true_map,
    )

    controls = get_angular_velocities_for_box(steps=60, radius_steps=8) * num_loops
    seen: set[int] = set()
    meas = np.array([])

    for control_turn_rate in controls:
        control = np.array([[1.0], [control_turn_rate]])
        sim.step(u=control)
        ekf.predict(u=control)

        pose_estimate = state_to_se3(ekf.x[0:6, 0])
        all_ids = [feature.id for feature in true_map.features]
        new_ids = [i for i in all_ids if i not in seen]
        if new_ids:
            new_features = [true_map.features[i] for i in new_ids]
            sightings = measure_distance_azimuth(
                state=sim.pose.as_vector(),
                features=new_features,
                noise=MEASUREMENT_NOISE,
            )
            for k, feature_id in enumerate(new_ids):
                dist, azi = sightings[2 * k, 0], sightings[2 * k + 1, 0]
                x0, y0 = initialize_landmark_estimate(pose_estimate, dist, azi)
                idx = 6 + 2 * feature_id
                ekf.x[idx, 0] = x0
                ekf.x[idx + 1, 0] = y0
                ekf.cov[idx : idx + 2, idx : idx + 2] = MEASUREMENT_NOISE * np.eye(2)
            seen.update(new_ids)

        observed = sorted(seen - set(new_ids))
        if observed:
            observed_features = [true_map.features[i] for i in observed]
            meas = measure_distance_azimuth(
                state=sim.pose.as_vector(),
                features=observed_features,
                noise=MEASUREMENT_NOISE,
            )
            ekf.update(
                z=meas,
                sensor=measure_distance_azimuth_slam,
                u=control,
                measurement_args=(observed, num_features),
            )

        estimated_features = [
            Feature(
                id=i, x=float(ekf.x[6 + 2 * i, 0]), y=float(ekf.x[6 + 2 * i + 1, 0])
            )
            for i in sorted(seen)
        ]
        sim.append_result(
            estimate=(state_to_se3(ekf.x[0:6, 0]), ekf.cov[:6, :6]),
            measurement=meas,
            show_plot=show_plot,
            estimated_features=estimated_features,
            measurement_stride=2,
        )

        if show_plot:
            x_str = np.array2string(ekf.x.T, precision=LOG_DECIMALS, floatmode="fixed")
            logger.info(f"SLAM estimate: {x_str}")

    return ekf, true_map, sim.pose


if __name__ == "__main__":
    pipeline(show_plot=True)
