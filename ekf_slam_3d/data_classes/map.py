"""Add a doc string to my files."""

from dataclasses import dataclass

import numpy as np
from loguru import logger

from config.definitions import MAP_DIM, MAP_NUM_FEATURES
from ekf_slam_3d.data_classes.lie_algebra import SE3


@dataclass
class Feature:
    """Dataclass to represent a detected feature."""

    id: int
    x: float
    y: float
    z: float = 0.0

    def as_vector(self) -> np.ndarray:
        """Return the feature as a 3-by-1 matrix."""
        return np.array([[self.x], [self.y], [self.z]])


class Map:
    """Dataclass to store features for a map."""

    def __init__(self, features: list[Feature] | None = None):
        if features is None:
            features = []
        self.features = features

    def append_feature(self, feature: Feature) -> None:
        """Append a feature to the map.

        :param feature: Feature to be appended
        :return: None
        """
        if self.feature_already_found(feature):
            logger.warning(f"Revisited landmark with I.D. {feature.id}.")
        else:
            self.features.append(feature)
            logger.info(f"Added landmark with I.D. {feature.id}.")

    def feature_already_found(self, new_feature: Feature) -> bool:
        """Check if the feature already exists.

        :param new_feature: Feature to check
        :return: True if the feature already exists, False otherwise
        """
        ids = [feature.id for feature in self.features]
        return new_feature.id in ids

    def update_feature_location(self, feature: Feature) -> None:
        """Update the location of a feature.

        :param feature: Feature to update
        :return: None
        """
        idx = [feature.id for feature in self.features].index(feature.id)
        self.features[idx].x = feature.x
        self.features[idx].y = feature.y


def make_random_map_planar(
    num_features: int = MAP_NUM_FEATURES, dim: tuple[int, int] = MAP_DIM
) -> Map:
    """Make a random map.

    :param num_features: Number of features to generate
    :param dim: Dimensions of the map
    :return: A map with random features
    """
    new_map = Map()
    for num in range(num_features):
        map_feature = Feature(
            id=num,
            x=np.random.uniform(0, dim[0]),
            y=np.random.uniform(0, dim[1]),
            z=0.0,
        )
        new_map.append_feature(map_feature)
    return new_map


def make_box_map_planar(
    num_features: int = MAP_NUM_FEATURES, dim: tuple[int, int] = MAP_DIM
) -> Map:
    """Make a box map with random points.

    :param num_features: Number of features to generate
    :param dim: Dimensions of the map
    :return: A map with random features
    """
    new_map = Map()

    num_features = 4 * round(num_features / 4)

    for num in range(0, int(num_features / 4)):
        x = np.random.uniform(0, dim[0])
        map_feature = Feature(id=num, x=x, y=0)
        new_map.append_feature(map_feature)
    for num in range(int(num_features / 4), int(num_features / 2)):
        y = np.random.uniform(0, dim[1])
        map_feature = Feature(id=num, x=dim[0], y=y)
        new_map.append_feature(map_feature)
    for num in range(int(num_features / 2), int(3 * num_features / 4)):
        x = np.random.uniform(0, dim[0])
        map_feature = Feature(id=num, x=x, y=dim[1])
        new_map.append_feature(map_feature)
    for num in range(int(3 * num_features / 4), num_features):
        y = np.random.uniform(0, dim[1])
        map_feature = Feature(id=num, x=0, y=y)
        new_map.append_feature(map_feature)

    return new_map


def distance_to_features(
    pose: SE3, features: list[Feature]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the distance between an SE3 pose and list of features.

    :param pose: SE3 pose
    :param features: List of features
    :return: (dx, dy, dz) - Distance vectors to each feature
    """
    dx = np.array([feature.x for feature in features]) - pose.x
    dy = np.array([feature.y for feature in features]) - pose.y
    dz = np.array([feature.z for feature in features]) - pose.z

    return dx, dy, dz
