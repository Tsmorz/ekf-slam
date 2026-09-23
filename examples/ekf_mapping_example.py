"""Basic docstring for my module."""

import numpy as np
from loguru import logger

from config.definitions import LANDMARK_INIT_VARIANCE, LOG_DECIMALS, MEASUREMENT_NOISE
from ekf_slam_3d.data_classes.lie_algebra import SE3
from ekf_slam_3d.data_classes.map import Feature, Map, make_box_map_planar
from ekf_slam_3d.data_classes.sensors import (
    initialize_landmark_estimate,
    measure_distance_azimuth,
    measure_distance_azimuth_map,
    step_dynamics,
    step_dynamics_map,
)
from ekf_slam_3d.modules.controller import get_angular_velocities_for_box
from ekf_slam_3d.modules.kalman_extended import ExtendedKalmanFilter
from ekf_slam_3d.modules.simulators import SlamSimulator
from ekf_slam_3d.modules.state_space import StateSpaceNonlinear


def pipeline(
    show_plot: bool, num_loops: int = 5
) -> tuple[ExtendedKalmanFilter, Map, SE3]:
    """Run the EKF for mapping with a known (ground-truth) trajectory.

    Localization is assumed solved here: the robot's true pose is used directly to
    take sightings, and the EKF only estimates the (x, y) location of each landmark.

    :param show_plot: whether to render the live simulation
    :param num_loops: number of times to drive the box trajectory around the map
    :return: (ekf holding the final map estimate, the true map, the final true pose)
    """
    logger.info("Running EKF pipeline for mapping.")

    true_map = make_box_map_planar(num_features=8, dim=(20, 20))
    num_features = len(true_map.features)

    ekf = ExtendedKalmanFilter(
        state_space_nonlinear=StateSpaceNonlinear(motion_model=step_dynamics_map),
        initial_x=np.zeros((2 * num_features, 1)),
        initial_covariance=LANDMARK_INIT_VARIANCE * np.eye(2 * num_features),
        process_noise=np.zeros((1, 1)),
        measurement_noise=MEASUREMENT_NOISE,
    )

    # the ground-truth pose is driven directly - the mapping EKF never sees it perturbed
    sim = SlamSimulator(
        state_space_nl=StateSpaceNonlinear(motion_model=step_dynamics),
        process_noise=np.array([[0.2, 0.0], [0.0, 0.04]]),
        initial_pose=SE3(xyz=np.array([10.0, 10.0, 0.0])),
        sim_map=true_map,
    )

    controls = get_angular_velocities_for_box(steps=60, radius_steps=8) * num_loops
    seen: set[int] = set()
    meas = np.array([])

    for control_turn_rate in controls:
        control = np.array([[1.0], [control_turn_rate]])
        true_pose = sim.step(u=control)

        all_ids = [feature.id for feature in true_map.features]
        new_ids = [i for i in all_ids if i not in seen]
        if new_ids:
            new_features = [true_map.features[i] for i in new_ids]
            sightings = measure_distance_azimuth(
                state=true_pose.as_vector(),
                features=new_features,
                noise=MEASUREMENT_NOISE,
            )
            for k, feature_id in enumerate(new_ids):
                dist, azi = sightings[2 * k, 0], sightings[2 * k + 1, 0]
                x0, y0 = initialize_landmark_estimate(true_pose, dist, azi)
                ekf.x[2 * feature_id, 0] = x0
                ekf.x[2 * feature_id + 1, 0] = y0
                idx = 2 * feature_id
                ekf.cov[idx : idx + 2, idx : idx + 2] = MEASUREMENT_NOISE * np.eye(2)
            seen.update(new_ids)

        observed = sorted(seen - set(new_ids))
        if observed:
            observed_features = [true_map.features[i] for i in observed]
            meas = measure_distance_azimuth(
                state=true_pose.as_vector(),
                features=observed_features,
                noise=MEASUREMENT_NOISE,
            )
            ekf.update(
                z=meas,
                sensor=measure_distance_azimuth_map,
                u=np.zeros((1, 1)),
                measurement_args=(true_pose, observed, num_features),
            )

        estimated_features = [
            Feature(id=i, x=float(ekf.x[2 * i, 0]), y=float(ekf.x[2 * i + 1, 0]))
            for i in sorted(seen)
        ]
        sim.append_result(
            estimate=(true_pose, 1e-6 * np.eye(6)),
            measurement=meas,
            show_plot=show_plot,
            estimated_features=estimated_features,
            measurement_stride=2,
        )

        if show_plot:
            x_str = np.array2string(ekf.x.T, precision=LOG_DECIMALS, floatmode="fixed")
            logger.info(f"map estimate: {x_str}")

    return ekf, true_map, sim.pose


if __name__ == "__main__":
    pipeline(show_plot=True)
