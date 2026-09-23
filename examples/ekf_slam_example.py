"""Basic docstring for my module."""

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from config.definitions import (
    CONTROL_NOISE_COVARIANCE,
    CONTROL_NOISE_COVARIANCE_3D,
    LANDMARK_INIT_VARIANCE,
    LOG_DECIMALS,
    MEASUREMENT_NOISE,
    ROBOT_SPEED,
    START_POSE_VARIANCE,
)
from ekf_slam_3d.data_classes.lie_algebra import SE3, state_to_se3
from ekf_slam_3d.data_classes.map import Map
from ekf_slam_3d.data_classes.sensors import (
    distance_azimuth_covariance,
    features_in_range,
    inverse_distance_azimuth_elevation_slam,
    inverse_distance_azimuth_slam,
    measure_distance_azimuth,
    measure_distance_azimuth_elevation,
    measure_distance_azimuth_elevation_slam,
    measure_distance_azimuth_slam,
    step_dynamics,
    step_dynamics_3d,
    step_dynamics_slam,
    step_dynamics_slam_3d,
)
from ekf_slam_3d.modules.controller import (
    pitch_rate_toward_path,
    pure_pursuit_turn_rate,
)
from ekf_slam_3d.modules.kalman_extended import ExtendedKalmanFilter, MeasurementSpec
from ekf_slam_3d.modules.simulators import LandmarkEstimate, SlamSimulator
from ekf_slam_3d.modules.state_space import StateSpaceNonlinear
from examples.slam_scenarios import SCENARIOS, SlamScenario


@dataclass(frozen=True)
class SlamModel:
    """The motion and sensor models for planar or 3D SLAM.

    Planar: landmarks are (x, y), sightings are [distance, azimuth], and the controls
    are [velocity, yaw rate]. 3D: landmarks are (x, y, z), sightings add elevation, and
    the controls add pitch rate. Readings per landmark always equal `landmark_dim`.
    """

    landmark_dim: int
    slam_motion: Callable
    robot_motion: Callable
    measure: Callable
    sensor: Callable
    inverse: Callable
    control_covariance: np.ndarray


PLANAR = SlamModel(
    landmark_dim=2,
    slam_motion=step_dynamics_slam,
    robot_motion=step_dynamics,
    measure=measure_distance_azimuth,
    sensor=measure_distance_azimuth_slam,
    inverse=inverse_distance_azimuth_slam,
    control_covariance=CONTROL_NOISE_COVARIANCE,
)
SPATIAL = SlamModel(
    landmark_dim=3,
    slam_motion=step_dynamics_slam_3d,
    robot_motion=step_dynamics_3d,
    measure=measure_distance_azimuth_elevation,
    sensor=measure_distance_azimuth_elevation_slam,
    inverse=inverse_distance_azimuth_elevation_slam,
    control_covariance=CONTROL_NOISE_COVARIANCE_3D,
)


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
    def landmark_dim(self) -> int:
        """Return the coordinates per landmark (2 planar, 3 in 3D)."""
        return self.scenario.landmark_dim

    @property
    def true_pose(self) -> SE3:
        """Return the final true pose."""
        return self.true_poses[-1]

    def _block(self, feature_id: int) -> slice:
        start = 6 + self.landmark_dim * feature_id
        return slice(start, start + self.landmark_dim)

    def landmark_estimate(self, feature_id: int) -> np.ndarray:
        """Return the current position estimate of a landmark."""
        return self.ekf.x[self._block(feature_id), 0]

    def landmark_covariance(self, feature_id: int) -> np.ndarray:
        """Return the current position covariance of a landmark."""
        block = self._block(feature_id)
        return self.ekf.cov[block, block]

    def landmark_truth(self, feature_id: int) -> np.ndarray:
        """Return a landmark's true position, in the same coordinates as its estimate."""
        feature = self.true_map.features[feature_id]
        return np.array([feature.x, feature.y, feature.z])[: self.landmark_dim]

    def landmark_for_plot(self, feature_id: int) -> LandmarkEstimate:
        """Return a landmark's current (x, y) estimate and covariance, for the plot."""
        estimate = self.landmark_estimate(feature_id)
        covariance = self.landmark_covariance(feature_id)[:2, :2]
        return LandmarkEstimate(float(estimate[0]), float(estimate[1]), covariance)

    def landmark_error(self, feature_id: int) -> float:
        """Return the distance between a landmark's estimate and its true position."""
        error = self.landmark_estimate(feature_id) - self.landmark_truth(feature_id)
        return float(np.linalg.norm(error))


def sense(
    run: SlamRun,
    model: SlamModel,
    true_pose: SE3,
    control: np.ndarray,
    seen: set[int],
) -> np.ndarray:
    """Take sightings of the landmarks in range and fold them into the EKF.

    Already-known landmarks correct the pose and map together; newly sighted ones are
    then seeded from the corrected pose, correlated with it.

    :param run: the run being recorded (holds the EKF and the true map)
    :param model: the planar or 3D models in use
    :param true_pose: where the robot really is (the sensor sees from here)
    :param control: the control applied on the step that got here
    :param seen: ids of landmarks already in the map - updated in place
    :return: stacked readings for every visible landmark
    """
    ekf, features = run.ekf, run.true_map.features
    stride = model.landmark_dim
    visible = features_in_range(true_pose, features, run.scenario.sensor_range)
    if not visible:
        return np.array([])

    readings = model.measure(
        state=true_pose.as_vector(),
        features=[features[i] for i in visible],
        noise=MEASUREMENT_NOISE,
    )
    reading = {
        i: readings[stride * k : stride * (k + 1)] for k, i in enumerate(visible)
    }

    known = [i for i in visible if i in seen]
    if known:
        z = np.vstack([reading[i] for i in known])
        ekf.update(
            z=z,
            sensor=model.sensor,
            u=control,
            measurement_args=(known, len(features)),
            spec=MeasurementSpec(
                covariance=distance_azimuth_covariance(z, MEASUREMENT_NOISE, stride)
            ),
        )

    for i in visible:
        if i not in seen:
            ekf.initialize_from_measurement(
                idx=6 + stride * i,
                z=reading[i],
                inverse_model=model.inverse,
                measurement_covariance=distance_azimuth_covariance(
                    reading[i], MEASUREMENT_NOISE, stride
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
    landmark coordinates (6:), x/y for planar scenarios and x/y/z for 3D ones. The robot
    only sees landmarks within sensor range, so the map is discovered as it drives, and
    it follows the scenario's loop (heading, plus altitude in 3D) using its own pose
    estimate.

    :param show_plot: whether to render the live simulation
    :param scenario: the synthetic dataset to run
    :return: the recorded run
    """
    logger.info(f"Running EKF pipeline for SLAM ({scenario.name}).")
    model = SPATIAL if scenario.three_d else PLANAR

    true_map = scenario.make_map()
    num_features = len(true_map.features)
    initial_pose, path = scenario.make_path()

    map_size = model.landmark_dim * num_features
    initial_x = np.vstack((initial_pose.as_vector(), np.zeros((map_size, 1))))
    initial_cov = START_POSE_VARIANCE * np.eye(6 + map_size)
    initial_cov[6:, 6:] = LANDMARK_INIT_VARIANCE * np.eye(map_size)

    ekf = ExtendedKalmanFilter(
        state_space_nonlinear=StateSpaceNonlinear(motion_model=model.slam_motion),
        initial_x=initial_x,
        initial_covariance=initial_cov,
        process_noise=model.control_covariance,
        measurement_noise=MEASUREMENT_NOISE**2,
    )
    sim = SlamSimulator(
        state_space_nl=StateSpaceNonlinear(motion_model=model.robot_motion),
        process_noise=model.control_covariance,
        initial_pose=initial_pose,
        sim_map=true_map,
        map_known=False,
    )

    run = SlamRun(ekf=ekf, true_map=true_map, scenario=scenario)
    seen: set[int] = set()
    control = np.zeros((len(model.control_covariance), 1))

    for step in range(scenario.num_loops * len(path) + 1):
        # sense before the first move, so the map is anchored to the known start pose
        if step > 0:
            pose_estimate = state_to_se3(ekf.x[0:6, 0])
            commands = [ROBOT_SPEED, pure_pursuit_turn_rate(pose_estimate, path)]
            if scenario.three_d:
                commands.append(pitch_rate_toward_path(pose_estimate, path))
            control = np.array(commands).reshape(-1, 1)
            sim.step(u=control)
            ekf.predict(u=control)

        meas = sense(run, model, sim.pose, control, seen)

        run.true_poses.append(sim.pose)
        run.estimated_poses.append(ekf.x[0:6, 0].copy())
        run.known_counts.append(len(seen))

        sim.append_result(
            estimate=(state_to_se3(ekf.x[0:6, 0]), ekf.cov[:6, :6]),
            measurement=meas,
            show_plot=show_plot,
            landmarks=[run.landmark_for_plot(i) for i in sorted(seen)],
            measurement_stride=model.landmark_dim,
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
