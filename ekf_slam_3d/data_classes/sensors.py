"""Add a doc string to my files."""

from collections.abc import Callable
from enum import Enum, auto

import numpy as np
from loguru import logger

from config.definitions import DELTA_T
from ekf_slam_3d.data_classes.lie_algebra import SE3, state_to_se3
from ekf_slam_3d.data_classes.map import Feature, distance_to_features


def measure_gps(
    state: np.ndarray,
    features: list[Feature],
    noise: np.ndarray | float | None = None,
) -> np.ndarray:
    """Calculate the gps position from a given pose.

    :param state: the current state vector
    :param noise: optional noise vector for measurement
    :return: the gps position
    """
    pose = state_to_se3(state)
    xyz = np.array([[pose.x], [pose.y], [pose.z]])
    if noise is not None:
        measurement_noise = np.random.normal(loc=0.0, scale=noise, size=xyz.shape)
        xyz = xyz + measurement_noise
    return xyz


def measure_azimuth(
    state: np.ndarray,
    features: list[Feature],
    noise: np.ndarray | float | None = None,
) -> np.ndarray:
    """Calculate the azimuth angle from a given pose to a list of features.

    :param state: the current state vector
    :param features: the list of features
    :param noise: optional noise vector for measurement
    :return: the azimuth angles
    """
    pose = state_to_se3(state)
    dx, dy, dz = distance_to_features(pose=pose, features=features)
    distance = np.linalg.norm(np.array([dx, dy, dz]), axis=0)
    azimuth = np.arctan2(dy, dx) - pose.yaw
    if noise is not None:
        measurement_noise = np.random.normal(
            loc=0.0, scale=noise / (1 + distance), size=dx.shape
        )
        azimuth = azimuth + measurement_noise

    return azimuth.reshape((len(azimuth), 1))


def measure_elevation(
    state: np.ndarray,
    features: list[Feature],
    noise: np.ndarray | float | None = None,
) -> np.ndarray:
    """Calculate the elevation angle from a given pose to a list of features.

    Elevation is measured above the horizontal plane (like azimuth, it is taken in a
    level, yaw-aligned sensor frame), so it lies in [-pi/2, pi/2].

    :param state: the current state vector
    :param features: the list of features
    :param noise: optional noise vector for measurement
    :return: the elevation angles
    """
    pose = state_to_se3(state)
    dx, dy, dz = distance_to_features(pose=pose, features=features)
    distance = np.linalg.norm(np.array([dx, dy, dz]), axis=0)
    elevation = np.arctan2(dz, np.hypot(dx, dy))
    if noise is not None:
        measurement_noise = np.random.normal(
            loc=0.0, scale=noise / (1 + distance), size=dx.shape
        )
        elevation = elevation + measurement_noise

    return elevation.reshape((len(elevation), 1))


def measure_distance(
    state: np.ndarray,
    features: list[Feature],
    noise: np.ndarray | float | None = None,
) -> np.ndarray:
    """Calculate the distance from a given pose to a list of features.

    :param state: the current state vector
    :param features: the list of features
    :param noise: optional noise vector for measurement
    :return: the distance
    """
    pose = state_to_se3(state)
    dx, dy, dz = distance_to_features(pose=pose, features=features)
    distance = np.linalg.norm(np.array([dx, dy, dz]), axis=0)
    if noise is not None:
        measurement_noise = np.random.normal(loc=0.0, scale=noise, size=dx.shape)
        distance = distance + measurement_noise

    return distance.reshape((len(distance), 1))


def measure_distance_azimuth_elevation(
    state: np.ndarray,
    features: list[Feature],
    noise: np.ndarray | float | None = None,
) -> np.ndarray:
    """Calculate the distance, azimuth, and elevation from a pose to a list of features.

    :param state: the current state vector
    :param features: the list of features
    :param noise: optional noise vector for measurement
    :return: stacked [distance, azimuth, elevation] measurements, one triple per feature
    """
    distance = measure_distance(state=state, features=features, noise=noise)
    azimuth = measure_azimuth(state=state, features=features, noise=noise)
    elevation = measure_elevation(state=state, features=features, noise=noise)

    merged = np.array((distance, azimuth, elevation)).T.ravel()
    return np.reshape(merged, (len(merged), 1))


def measure_distance_azimuth(
    state: np.ndarray,
    features: list[Feature],
    noise: np.ndarray | float | None = None,
) -> np.ndarray:
    """Calculate the planar distance and azimuth from a given pose to a list of features.

    Used for the mapping and SLAM pipelines, which track features as (x, y) pairs.

    :param state: the current state vector
    :param features: the list of features
    :param noise: optional noise vector for measurement
    :return: stacked [distance, azimuth] measurements, one pair per feature
    """
    distance = measure_distance(state=state, features=features, noise=noise)
    azimuth = measure_azimuth(state=state, features=features, noise=noise)

    merged = np.array((distance, azimuth)).T.ravel()
    return np.reshape(merged, (len(merged), 1))


def distance_azimuth_covariance(
    readings: np.ndarray, noise: float, stride: int = 2
) -> np.ndarray:
    """Return the covariance of the noise the range/bearing sensors add to `readings`.

    `noise` is a standard deviation: distance gets `noise`, and each angle (azimuth,
    and elevation when `stride` is 3) gets `noise / (1 + distance)`. The simulator
    scales by the true distance, which a filter can't know, so the measured distance
    stands in for it here.

    :param readings: stacked [distance, azimuth] pairs, or [distance, azimuth,
        elevation] triples when `stride` is 3
    :param noise: the noise scale passed to the measurement function
    :param stride: values per landmark (2 or 3)
    :return: diagonal covariance approximating the simulated sensor's
    """
    variances = np.empty(len(readings))
    variances[0::stride] = noise**2
    for offset in range(1, stride):
        variances[offset::stride] = (noise / (1 + readings[0::stride, 0])) ** 2
    return np.diag(variances)


def angle_mask(length: int, stride: int, offsets: tuple[int, ...]) -> np.ndarray:
    """Build a boolean mask marking the angle-valued entries of an interleaved measurement.

    `ExtendedKalmanFilter.update()` wraps these innovation entries to [-pi, pi)
    instead of using the raw (possibly ~2*pi) difference.

    :param length: total length of the measurement vector
    :param stride: number of values per group (e.g. 2 for [distance, azimuth] pairs)
    :param offsets: positions within each group that are angle-valued (e.g. (1,) for azimuth)
    :return: boolean mask, True at angle-valued indices
    """
    mask = np.zeros(length, dtype=bool)
    for offset in offsets:
        mask[offset::stride] = True
    return mask


def initialize_landmark_estimate(
    pose: SE3, distance: float, azimuth: float
) -> tuple[float, float]:
    """Back out a landmark's global (x, y) position from a single range-azimuth sighting.

    Used to seed a landmark's EKF state on first observation, rather than starting it
    at the origin - which would make the azimuth/distance Jacobian a poor local
    approximation until the filter had a chance to converge.

    :param pose: the pose the sighting was taken from
    :param distance: measured distance to the landmark
    :param azimuth: measured azimuth (bearing) to the landmark, relative to pose.yaw
    :return: (x, y) estimate of the landmark position in the global frame
    """
    heading = pose.yaw + azimuth
    x = pose.x + distance * np.cos(heading)
    y = pose.y + distance * np.sin(heading)
    return x, y


def initialize_landmark_estimate_3d(
    pose: SE3, distance: float, azimuth: float, elevation: float
) -> tuple[float, float, float]:
    """Back out a landmark's global (x, y, z) from one range-azimuth-elevation sighting.

    :param pose: the pose the sighting was taken from
    :param distance: measured distance to the landmark
    :param azimuth: measured azimuth, relative to pose.yaw
    :param elevation: measured elevation above the horizontal plane
    :return: (x, y, z) estimate of the landmark position in the global frame
    """
    heading = pose.yaw + azimuth
    horizontal = distance * np.cos(elevation)
    return (
        pose.x + horizontal * np.cos(heading),
        pose.y + horizontal * np.sin(heading),
        pose.z + distance * np.sin(elevation),
    )


def inverse_distance_azimuth_elevation_slam(
    state_measurement: np.ndarray,
) -> np.ndarray:
    """Inverse model for 3D SLAM: [pose(6); landmarks; d; azimuth; elevation] -> (x, y, z).

    :param state_measurement: SLAM state stacked with one [distance; azimuth; elevation]
    :return: (3, 1) landmark position estimate in the global frame
    """
    pose = state_to_se3(state_measurement[0:6, 0])
    distance, azimuth, elevation = state_measurement[-3:, 0]
    x, y, z = initialize_landmark_estimate_3d(pose, distance, azimuth, elevation)
    return np.array([[x], [y], [z]])


def inverse_distance_azimuth_slam(state_measurement: np.ndarray) -> np.ndarray:
    """Inverse range-azimuth model for SLAM: [pose(6); landmarks; distance; azimuth] -> (x, y).

    Reads the pose from the state so `ExtendedKalmanFilter.initialize_from_measurement`
    can differentiate the seeded landmark with respect to the pose it was sighted from.

    :param state_measurement: SLAM state stacked with a single [distance; azimuth] sighting
    :return: (2, 1) landmark position estimate in the global frame
    """
    pose = state_to_se3(state_measurement[0:6, 0])
    distance, azimuth = state_measurement[-2, 0], state_measurement[-1, 0]
    x, y = initialize_landmark_estimate(pose, distance, azimuth)
    return np.array([[x], [y]])


def inverse_distance_azimuth_map(
    state_measurement: np.ndarray, args: tuple[SE3]
) -> np.ndarray:
    """Inverse range-azimuth model for mapping with a known pose: sighting -> (x, y).

    :param state_measurement: landmark state stacked with a single [distance; azimuth]
    :param args: (observer pose,)
    :return: (2, 1) landmark position estimate in the global frame
    """
    (pose,) = args
    distance, azimuth = state_measurement[-2, 0], state_measurement[-1, 0]
    x, y = initialize_landmark_estimate(pose, distance, azimuth)
    return np.array([[x], [y]])


def features_in_range(
    pose: SE3, features: list[Feature], max_range: float
) -> list[int]:
    """Return the ids of the features within `max_range` of the pose.

    :param pose: the observer pose
    :param features: candidate features
    :param max_range: sensing range
    :return: ids of the visible features
    """
    return [
        feature.id
        for feature in features
        if np.linalg.norm([feature.x - pose.x, feature.y - pose.y, feature.z - pose.z])
        <= max_range
    ]


def _landmark_xy(
    state: np.ndarray, num_landmarks: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split a state vector's leading interleaved x/y landmark pairs into (xs, ys).

    Only the first ``2 * num_landmarks`` rows are used, so this is safe to call on a
    state vector that has extra rows (e.g. a control input) stacked after the landmarks.

    :param state: state vector with interleaved x/y landmark pairs in its first rows
    :param num_landmarks: total number of landmarks packed into the state
    :return: (xs, ys) arrays of landmark coordinates
    """
    landmarks = state[: 2 * num_landmarks, 0]
    return landmarks[0::2], landmarks[1::2]


def measure_distance_azimuth_map(
    state: np.ndarray,
    args: tuple[SE3, list[int], int],
    noise: np.ndarray | float | None = None,
) -> np.ndarray:
    """Range-azimuth measurement model for mapping with a known pose.

    :param state: landmark position estimate vector (interleaved x/y pairs)
    :param args: (observer pose, observed feature ids, total number of landmarks)
    :param noise: optional noise scale for the measurement
    :return: stacked [distance, azimuth] measurements for the observed ids
    """
    pose, feature_ids, num_landmarks = args
    xs, ys = _landmark_xy(state, num_landmarks)
    ids = np.array(feature_ids)
    dx, dy = xs[ids] - pose.x, ys[ids] - pose.y
    distance = np.sqrt(dx**2 + dy**2)
    azimuth = np.arctan2(dy, dx) - pose.yaw
    if noise is not None:
        distance = distance + np.random.normal(
            loc=0.0, scale=noise, size=distance.shape
        )
        azimuth = azimuth + np.random.normal(
            loc=0.0, scale=noise / (1 + distance), size=azimuth.shape
        )

    merged = np.array((distance, azimuth)).T.ravel()
    return np.reshape(merged, (len(merged), 1))


def step_dynamics_map(state_control: np.ndarray, dt: float = DELTA_T) -> np.ndarray:
    """Return the landmark state unchanged - landmarks do not evolve on their own.

    :param state_control: the landmark state stacked with a (unused) dummy control
    :param dt: the time step (unused, kept for interface parity with other motion models)
    :return: the landmark state vector, unchanged
    """
    del dt
    return state_control[:-1, 0:1]


def measure_distance_azimuth_slam(
    state: np.ndarray,
    args: tuple[list[int], int],
    noise: np.ndarray | float | None = None,
) -> np.ndarray:
    """Range-azimuth measurement model for full SLAM (pose and map both in the state).

    :param state: SLAM state vector - pose (0:6) followed by interleaved x/y landmark pairs
    :param args: (observed feature ids, total number of landmarks)
    :param noise: optional noise scale for the measurement
    :return: stacked [distance, azimuth] measurements for the observed ids
    """
    feature_ids, num_landmarks = args
    pose = state_to_se3(state[0:6, 0])
    xs, ys = _landmark_xy(state[6:, :], num_landmarks)
    ids = np.array(feature_ids)
    dx, dy = xs[ids] - pose.x, ys[ids] - pose.y
    distance = np.sqrt(dx**2 + dy**2)
    azimuth = np.arctan2(dy, dx) - pose.yaw
    if noise is not None:
        distance = distance + np.random.normal(
            loc=0.0, scale=noise, size=distance.shape
        )
        azimuth = azimuth + np.random.normal(
            loc=0.0, scale=noise / (1 + distance), size=azimuth.shape
        )

    merged = np.array((distance, azimuth)).T.ravel()
    return np.reshape(merged, (len(merged), 1))


def measure_distance_azimuth_elevation_slam(
    state: np.ndarray,
    args: tuple[list[int], int],
    noise: np.ndarray | float | None = None,
) -> np.ndarray:
    """Range-azimuth-elevation model for 3D SLAM (pose and x/y/z landmarks in the state).

    :param state: SLAM state vector - pose (0:6) followed by interleaved x/y/z triples
    :param args: (observed feature ids, total number of landmarks)
    :param noise: optional noise scale for the measurement
    :return: stacked [distance, azimuth, elevation] measurements for the observed ids
    """
    feature_ids, num_landmarks = args
    pose = state_to_se3(state[0:6, 0])
    landmarks = state[6 : 6 + 3 * num_landmarks, 0].reshape(num_landmarks, 3)
    ids = np.array(feature_ids)
    dx = landmarks[ids, 0] - pose.x
    dy = landmarks[ids, 1] - pose.y
    dz = landmarks[ids, 2] - pose.z
    horizontal = np.hypot(dx, dy)
    distance = np.hypot(horizontal, dz)
    azimuth = np.arctan2(dy, dx) - pose.yaw
    elevation = np.arctan2(dz, horizontal)
    if noise is not None:
        angle_scale = noise / (1 + distance)
        distance = distance + np.random.normal(0.0, noise, size=distance.shape)
        azimuth = azimuth + np.random.normal(0.0, angle_scale, size=azimuth.shape)
        elevation = elevation + np.random.normal(0.0, angle_scale, size=elevation.shape)

    merged = np.array((distance, azimuth, elevation)).T.ravel()
    return np.reshape(merged, (len(merged), 1))


def step_dynamics_slam(state_control: np.ndarray, dt: float = DELTA_T) -> np.ndarray:
    """Motion model for full SLAM: the pose evolves per `step_dynamics`, landmarks are static.

    :param state_control: pose, landmarks, and control vectors stacked together
    :param dt: the time step
    :return: the next SLAM state vector
    """
    num_landmark_dims = state_control.shape[0] - 6 - 2
    landmarks = state_control[6 : 6 + num_landmark_dims, 0:1]
    pose_vec = step_dynamics(np.vstack((state_control[:6], state_control[-2:])), dt=dt)
    return np.vstack((pose_vec, landmarks))


def step_dynamics(state_control: np.ndarray, dt: float = DELTA_T) -> np.ndarray:
    """Define the equations of motion for controls [velocity, yaw rate].

    :param state_control: the pose (6) stacked with the 2 controls
    :param dt: the time step
    :return: the pose after applying the motion equations
    """
    no_pitch_rate = np.zeros((1, 1))
    return step_dynamics_3d(
        np.vstack((state_control[:6], state_control[-2:], no_pitch_rate)), dt=dt
    )


def step_dynamics_3d(state_control: np.ndarray, dt: float = DELTA_T) -> np.ndarray:
    """Define the equations of motion for controls [velocity, yaw rate, pitch rate].

    The vehicle moves along its heading, climbing or descending with its pitch.

    :param state_control: the pose (6) stacked with the 3 controls
    :param dt: the time step
    :return: the pose after applying the motion equations
    """
    vel, yaw_rate, pitch_rate = state_control[-3:, 0]
    pose = state_to_se3(state_control[:6, 0])

    state_vec = np.zeros((6, 1))
    state_vec[0, 0] = pose.x + vel * dt * np.cos(pose.yaw) * np.cos(pose.pitch)
    state_vec[1, 0] = pose.y + vel * dt * np.sin(pose.yaw) * np.cos(pose.pitch)
    state_vec[2, 0] = pose.z + vel * dt * np.sin(pose.pitch)
    state_vec[3, 0] = pose.roll
    state_vec[4, 0] = pose.pitch + pitch_rate * dt
    state_vec[5, 0] = pose.yaw + yaw_rate * dt
    return state_vec


def step_dynamics_slam_3d(state_control: np.ndarray, dt: float = DELTA_T) -> np.ndarray:
    """Motion model for 3D SLAM: the pose evolves per `step_dynamics_3d`, landmarks are static.

    :param state_control: pose, x/y/z landmarks, and the 3 controls stacked together
    :param dt: the time step
    :return: the next SLAM state vector
    """
    landmarks = state_control[6:-3, 0:1]
    pose_vec = step_dynamics_3d(
        np.vstack((state_control[:6], state_control[-3:])), dt=dt
    )
    return np.vstack((pose_vec, landmarks))


class Sensor(Enum):
    """Define the individual sensor types."""

    def __init__(self, code: int, func: Callable):
        if func is None:
            logger.error("Not implemented.")
        self.code = code
        self.func = func

    GPS = auto(), measure_gps
    IMU = auto(), measure_distance
    DIST = auto(), measure_distance
    ELE = auto(), measure_elevation
    AZI = auto(), measure_azimuth
    DIST_AZI_ELE = auto(), measure_distance_azimuth_elevation


# (stride, angle-valued offsets within each group) of each measurement function's output
ANGLE_LAYOUTS: dict[Callable, tuple[int, tuple[int, ...]]] = {
    measure_azimuth: (1, (0,)),
    measure_elevation: (1, (0,)),
    measure_distance_azimuth: (2, (1,)),
    measure_distance_azimuth_map: (2, (1,)),
    measure_distance_azimuth_slam: (2, (1,)),
    measure_distance_azimuth_elevation: (3, (1, 2)),
    measure_distance_azimuth_elevation_slam: (3, (1, 2)),
}


def default_angle_mask(sensor: Callable, length: int) -> np.ndarray | None:
    """Return the angle mask for a known measurement function's output, if it has angles.

    :param sensor: the measurement function
    :param length: length of the measurement vector
    :return: boolean mask of angle-valued entries, or None if the sensor has none
    """
    layout = ANGLE_LAYOUTS.get(sensor)
    if layout is None:
        return None
    stride, offsets = layout
    return angle_mask(length, stride, offsets)
