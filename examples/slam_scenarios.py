"""Synthetic SLAM datasets, each built to stress a different way the filter can break."""

from dataclasses import dataclass

import numpy as np

from config.definitions import LANDMARK_JITTER_FRACTION, SENSOR_RANGE
from ekf_slam_3d.data_classes.lie_algebra import SE3
from ekf_slam_3d.data_classes.map import Feature, Map
from ekf_slam_3d.modules.controller import box_path


@dataclass(frozen=True)
class SlamScenario:
    """A landmark map plus a closed loop for the robot to drive around it."""

    name: str
    description: str
    num_landmarks: int = 12
    map_size: float = 30.0
    map_center: tuple[float, float] = (15.0, 15.0)
    heading: float = 0.0
    turn: int = 1
    side_steps: int = 12
    radius_steps: int = 6
    num_loops: int = 3
    sensor_range: float = SENSOR_RANGE
    # 3D: landmarks at heights drawn from `landmark_heights`, and a path whose altitude
    # swings `altitude_swing` above and below `altitude` once per loop
    three_d: bool = False
    landmark_heights: tuple[float, float] = (0.0, 0.0)
    altitude: float = 0.0
    altitude_swing: float = 0.0

    @property
    def landmark_dim(self) -> int:
        """Return the coordinates per landmark in the EKF state (2 planar, 3 in 3D)."""
        return 3 if self.three_d else 2

    def make_map(self) -> Map:
        """Return landmarks spaced (with jitter) around the perimeter of the map square.

        Purely random placement can leave a landmark out of reach, or leave the start
        pose with nothing in view to anchor the map to; even spacing avoids both.
        """
        size = self.map_size
        spacing = 4 * size / self.num_landmarks
        jitter_range = LANDMARK_JITTER_FRACTION * spacing
        jitter = np.random.uniform(-jitter_range, jitter_range, self.num_landmarks)
        arc = ((np.arange(self.num_landmarks) + 0.5) * spacing + jitter) % (4 * size)
        corner = np.array([[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]])
        direction = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        side = (arc // size).astype(int)
        points = corner[side] + direction[side] * (arc % size)[:, None]
        points += np.array(self.map_center) - size / 2
        heights = np.zeros(self.num_landmarks)
        if self.three_d:
            heights = np.random.uniform(*self.landmark_heights, self.num_landmarks)
        return Map(
            [
                Feature(id=i, x=float(x), y=float(y), z=float(z))
                for i, ((x, y), z) in enumerate(zip(points, heights, strict=True))
            ]
        )

    def make_path(self) -> tuple[SE3, np.ndarray]:
        """Return the loop's start pose and (N, 3) positions, centered on the map."""
        rpy = np.array([0.0, 0.0, self.heading])
        path = box_path(
            SE3(roll_pitch_yaw=rpy), self.side_steps, self.radius_steps, turn=self.turn
        )
        offset = np.array(self.map_center) - (path.min(axis=0) + path.max(axis=0)) / 2
        phase = 2 * np.pi * np.arange(len(path)) / len(path)
        altitude = self.altitude + self.altitude_swing * np.sin(phase)
        start = SE3(
            xyz=np.array([offset[0], offset[1], altitude[0]]), roll_pitch_yaw=rpy
        )
        return start, np.column_stack((path + offset, altitude))


SCENARIOS: dict[str, SlamScenario] = {
    scenario.name: scenario
    for scenario in [
        SlamScenario(
            name="baseline",
            description="Counterclockwise loops inside a ring of 12 landmarks. Only "
            "landmarks within sensor range are seen, so the map is discovered as the "
            "robot drives and tightened on every loop closure.",
        ),
        SlamScenario(
            name="clockwise",
            description="Clockwise loops - catches sign/handedness bugs in turning, "
            "bearings, or landmark initialization.",
            turn=-1,
        ),
        SlamScenario(
            name="branch_cut",
            description="Starts facing -x, so yaw and bearings sit on the +-pi atan2 "
            "branch cut - catches unwrapped angle innovations.",
            heading=np.pi,
        ),
        SlamScenario(
            name="long_run",
            description="Ten loops, so yaw accumulates to dozens of radians - catches "
            "anything that assumes yaw stays within +-pi.",
            num_loops=10,
        ),
        SlamScenario(
            name="sparse",
            description="Six landmarks, so stretches of the loop are driven with nothing "
            "in view - catches landmarks seeded from a drifted pose not being pulled "
            "back on loop closure.",
            num_landmarks=6,
        ),
        SlamScenario(
            name="dense",
            description="24 landmarks, many in view at once - catches indexing/stride "
            "bugs in the interleaved [distance, azimuth] measurement layout.",
            num_landmarks=24,
        ),
        SlamScenario(
            name="offset_map",
            description="The whole world sits far from the origin in negative x - "
            "catches code that assumes positions near (0, 0).",
            map_center=(-80.0, 45.0),
        ),
        SlamScenario(
            name="baseline_3d",
            description="3D: landmarks at heights of 0-8 m, and the robot climbs and "
            "descends between 2 and 6 m every loop, sensing range, azimuth, and "
            "elevation - pose and map are estimated in x, y, and z.",
            three_d=True,
            landmark_heights=(0.0, 8.0),
            altitude=4.0,
            altitude_swing=2.0,
        ),
        SlamScenario(
            name="steep_3d",
            description="3D with a tall map (0-12 m) and a 1-9 m altitude swing - "
            "steep climbs and large elevation angles stress pitch and elevation.",
            three_d=True,
            landmark_heights=(0.0, 12.0),
            sensor_range=13.0,
            altitude=5.0,
            altitude_swing=4.0,
        ),
        SlamScenario(
            name="branch_cut_3d",
            description="3D, starting facing -x so yaw and bearings sit on the +-pi "
            "branch cut while elevation is also being estimated.",
            three_d=True,
            heading=np.pi,
            landmark_heights=(0.0, 8.0),
            altitude=4.0,
            altitude_swing=2.0,
        ),
    ]
}
