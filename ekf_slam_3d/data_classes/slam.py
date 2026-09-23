"""Basic docstring for my module."""

from dataclasses import dataclass, field

import numpy as np

from ekf_slam_3d.data_classes.lie_algebra import SE3
from ekf_slam_3d.data_classes.map import Feature, Map


@dataclass
class PoseMap:
    """Class representing the pose of the vehicle and the map."""

    pose: SE3 = field(default_factory=SE3)
    map: Map = field(default_factory=Map)

    def as_vector(self) -> np.ndarray:
        """Return the state data as a vector: pose (6) plus interleaved x/y feature pairs."""
        feature_vector = np.zeros((2 * len(self.map.features), 1))
        for idx, feature in enumerate(self.map.features):
            feature_vector[2 * idx, 0] = feature.x
            feature_vector[2 * idx + 1, 0] = feature.y
        return np.vstack((self.pose.as_vector(), feature_vector))

    def from_vector(self, ekf_state: np.ndarray) -> None:
        """Convert the state vector to the corresponding attributes.

        :param ekf_state: The state vector to convert
        :return: None
        """
        # update the pose
        xyz = ekf_state[0:3]
        rpy = ekf_state[3:6]
        self.pose = SE3(xyz=xyz, roll_pitch_yaw=rpy)

        # update the map features
        features_x = ekf_state[6::2, 0]
        features_y = ekf_state[7::2, 0]
        features = [
            Feature(x=float(x), y=float(y), id=idx)
            for idx, (x, y) in enumerate(zip(features_x, features_y, strict=False))
        ]
        self.map = Map(features=features)

    @property
    def num_features(self) -> int:
        """Return the number of features tracked in the map."""
        return len(self.map.features)
