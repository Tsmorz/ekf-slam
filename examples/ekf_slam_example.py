"""Basic docstring for my module."""

import argparse
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from config.definitions import (
    CONTROL_NOISE_COVARIANCE,
    LANDMARK_INIT_VARIANCE,
    LOG_DECIMALS,
    MEASUREMENT_NOISE,
    START_POSE_VARIANCE,
)
from ekf_slam_3d.data_classes.lie_algebra import SE3, state_to_se3
from ekf_slam_3d.data_classes.map import Map
from ekf_slam_3d.data_classes.sensors import (
    angle_mask,
    distance_azimuth_covariance,
    features_in_range,
    inverse_distance_azimuth_slam,
    measure_distance_azimuth,
    measure_distance_azimuth_slam,
    step_dynamics,
    step_dynamics_slam,
)
from ekf_slam_3d.modules.controller import pure_pursuit_turn_rate
from ekf_slam_3d.modules.kalman_extended import ExtendedKalmanFilter, MeasurementSpec
from ekf_slam_3d.modules.simulators import LandmarkEstimate, SlamSimulator
from ekf_slam_3d.modules.state_space import StateSpaceNonlinear
from examples.slam_scenarios import SCENARIOS, SlamScenario

SPEED = 1.0


@dataclass
class SlamRun:
    """What a SLAM run produced, recorded at every sensing step (index 0 = the start)."""

    ekf: ExtendedKalmanFilter
    true_map: Map
    scenario: SlamScenario
    true_poses: list[SE3] = field(default_factory=list)
    estimated_poses: list[np.ndarray] = field(default_factory=list)
    known_counts: list[int] = field(default_factory=list)
    # landmark id -> (position error, covariance trace) right after it was seeded
    first_seen: dict[int, tuple[float, float]] = field(default_factory=dict)

    @property
    def true_pose(self) -> SE3:
        """Return the final true pose."""
        return self.true_poses[-1]

    def landmark_estimate(self, feature_id: int) -> np.ndarray:
        """Return the current (x, y) estimate of a landmark."""
        idx = 6 + 2 * feature_id
        return self.ekf.x[idx : idx + 2, 0]

    def landmark_covariance(self, feature_id: int) -> np.ndarray:
        """Return the current 2x2 position covariance of a landmark."""
        idx = 6 + 2 * feature_id
        return self.ekf.cov[idx : idx + 2, idx : idx + 2]

    def landmark_for_plot(self, feature_id: int) -> LandmarkEstimate:
        """Return a landmark's current estimate and covariance, for the live plot."""
        x, y = self.landmark_estimate(feature_id)
        return LandmarkEstimate(
            float(x), float(y), self.landmark_covariance(feature_id)
        )

    def landmark_error(self, feature_id: int) -> float:
        """Return the distance between a landmark's estimate and its true position."""
        feature = self.true_map.features[feature_id]
        est = self.landmark_estimate(feature_id)
        return float(np.hypot(est[0] - feature.x, est[1] - feature.y))


def sense(
    run: SlamRun,
    true_pose: SE3,
    control: np.ndarray,
    seen: set[int],
    sensor_range: float,
) -> np.ndarray:
    """Take range-azimuth sightings of the landmarks in range and fold them into the EKF.

    Already-known landmarks correct the pose and map together; newly sighted ones are
    then seeded from the corrected pose, correlated with it.

    :param run: the run being recorded (holds the EKF and the true map)
    :param true_pose: where the robot really is (the sensor sees from here)
    :param control: the control applied on the step that got here
    :param seen: ids of landmarks already in the map - updated in place
    :param sensor_range: how far the sensor can see
    :return: stacked [distance, azimuth] readings for every visible landmark
    """
    ekf, features = run.ekf, run.true_map.features
    visible = features_in_range(true_pose, features, sensor_range)
    if not visible:
        return np.array([])

    readings = measure_distance_azimuth(
        state=true_pose.as_vector(),
        features=[features[i] for i in visible],
        noise=MEASUREMENT_NOISE,
    )
    reading = {i: readings[2 * k : 2 * k + 2] for k, i in enumerate(visible)}

    known = [i for i in visible if i in seen]
    if known:
        z = np.vstack([reading[i] for i in known])
        ekf.update(
            z=z,
            sensor=measure_distance_azimuth_slam,
            u=control,
            measurement_args=(known, len(features)),
            spec=MeasurementSpec(
                covariance=distance_azimuth_covariance(z, MEASUREMENT_NOISE),
                angle_mask=angle_mask(len(z), stride=2, offsets=(1,)),
            ),
        )

    for i in visible:
        if i not in seen:
            ekf.initialize_from_measurement(
                idx=6 + 2 * i,
                z=reading[i],
                inverse_model=inverse_distance_azimuth_slam,
                measurement_covariance=distance_azimuth_covariance(
                    reading[i], MEASUREMENT_NOISE
                ),
            )
            seen.add(i)
            trace = float(np.trace(run.landmark_covariance(i)))
            run.first_seen[i] = (run.landmark_error(i), trace)
    return readings


def pipeline(
    show_plot: bool, scenario: SlamScenario = SCENARIOS["baseline"]
) -> SlamRun:
    """Run the EKF for full SLAM: the pose and the map are estimated together.

    Neither is known ahead of time - the EKF state packs the pose (0:6) and interleaved
    x/y landmark pairs (6:). The robot only sees landmarks within sensor range, so the
    map is discovered as it drives, and it steers around the scenario's loop using its
    own pose estimate.

    :param show_plot: whether to render the live simulation
    :param scenario: the synthetic dataset to run
    :return: the recorded run
    """
    logger.info(f"Running EKF pipeline for SLAM ({scenario.name}).")

    true_map = scenario.make_map()
    num_features = len(true_map.features)
    initial_pose, path = scenario.make_path()

    initial_x = np.vstack((initial_pose.as_vector(), np.zeros((2 * num_features, 1))))
    initial_cov = START_POSE_VARIANCE * np.eye(6 + 2 * num_features)
    initial_cov[6:, 6:] = LANDMARK_INIT_VARIANCE * np.eye(2 * num_features)

    ekf = ExtendedKalmanFilter(
        state_space_nonlinear=StateSpaceNonlinear(motion_model=step_dynamics_slam),
        initial_x=initial_x,
        initial_covariance=initial_cov,
        process_noise=CONTROL_NOISE_COVARIANCE,
        measurement_noise=MEASUREMENT_NOISE,
    )
    sim = SlamSimulator(
        state_space_nl=StateSpaceNonlinear(motion_model=step_dynamics),
        process_noise=CONTROL_NOISE_COVARIANCE,
        initial_pose=initial_pose,
        sim_map=true_map,
        map_known=False,
    )

    run = SlamRun(ekf=ekf, true_map=true_map, scenario=scenario)
    seen: set[int] = set()
    control = np.zeros((2, 1))

    for step in range(scenario.num_loops * len(path) + 1):
        # sense before the first move, so the map is anchored to the known start pose
        if step > 0:
            pose_estimate = state_to_se3(ekf.x[0:6, 0])
            turn_rate = pure_pursuit_turn_rate(pose_estimate, path, speed=SPEED)
            control = np.array([[SPEED], [turn_rate]])
            sim.step(u=control)
            ekf.predict(u=control)

        meas = sense(run, sim.pose, control, seen, scenario.sensor_range)

        run.true_poses.append(sim.pose)
        run.estimated_poses.append(ekf.x[0:6, 0].copy())
        run.known_counts.append(len(seen))

        sim.append_result(
            estimate=(state_to_se3(ekf.x[0:6, 0]), ekf.cov[:6, :6]),
            measurement=meas,
            show_plot=show_plot,
            landmarks=[run.landmark_for_plot(i) for i in sorted(seen)],
            measurement_stride=2,
        )

        if show_plot:
            x_str = np.array2string(ekf.x.T, precision=LOG_DECIMALS, floatmode="fixed")
            logger.info(f"SLAM estimate: {x_str}")

    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="baseline",
        help="which synthetic dataset to run",
    )
    pipeline(show_plot=True, scenario=SCENARIOS[parser.parse_args().scenario])
