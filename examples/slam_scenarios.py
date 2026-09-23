"""Synthetic SLAM datasets, each built to stress a different way the filter can break."""

from dataclasses import dataclass

import numpy as np

from config.definitions import SENSOR_RANGE
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

    def make_map(self) -> Map:
        """Return landmarks spaced (with jitter) around the perimeter of the map square.

        Purely random placement can leave a landmark out of reach, or leave the start
        pose with nothing in view to anchor the map to; even spacing avoids both.
        """
        size = self.map_size
        spacing = 4 * size / self.num_landmarks
        jitter = np.random.uniform(-0.3, 0.3, self.num_landmarks) * spacing
        arc = ((np.arange(self.num_landmarks) + 0.5) * spacing + jitter) % (4 * size)
        corner = np.array([[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]])
        direction = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        side = (arc // size).astype(int)
        points = corner[side] + direction[side] * (arc % size)[:, None]
        points += np.array(self.map_center) - size / 2
        return Map(
            [Feature(id=i, x=float(x), y=float(y)) for i, (x, y) in enumerate(points)]
        )

    def make_path(self) -> tuple[SE3, np.ndarray]:
        """Return the loop's start pose and (N, 2) positions, centered on the map."""
        rpy = np.array([0.0, 0.0, self.heading])
        path = box_path(
            SE3(roll_pitch_yaw=rpy), self.side_steps, self.radius_steps, turn=self.turn
        )
        offset = np.array(self.map_center) - (path.min(axis=0) + path.max(axis=0)) / 2
        start = SE3(xyz=np.array([offset[0], offset[1], 0.0]), roll_pitch_yaw=rpy)
        return start, path + offset


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
    ]
}
