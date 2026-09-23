"""Add a doc string to my files."""

import numpy as np

from ekf_slam_3d.data_classes.lie_algebra import SE3
from ekf_slam_3d.data_classes.map import Feature, Map
from ekf_slam_3d.data_classes.slam import PoseMap


def test_pose_map_as_vector() -> None:
    """Test that a PoseMap packs pose and features into a single vector."""
    # Arrange
    pose = SE3(xyz=np.array([1.0, 2.0, 3.0]), roll_pitch_yaw=np.array([0.1, 0.2, 0.3]))
    features = [Feature(id=0, x=4.0, y=5.0), Feature(id=1, x=6.0, y=7.0)]
    pose_map = PoseMap(pose=pose, map=Map(features=features))

    # Act
    vector = pose_map.as_vector()

    # Assert
    expected = np.array(
        [[1.0], [2.0], [3.0], [0.1], [0.2], [0.3], [4.0], [5.0], [6.0], [7.0]]
    )
    np.testing.assert_array_almost_equal(vector, expected)


def test_pose_map_as_vector_no_features() -> None:
    """Test that a PoseMap with no features packs to just the pose vector."""
    # Arrange
    pose = SE3(xyz=np.array([1.0, 2.0, 3.0]), roll_pitch_yaw=np.array([0.1, 0.2, 0.3]))
    pose_map = PoseMap(pose=pose)

    # Act
    vector = pose_map.as_vector()

    # Assert
    assert vector.shape == (6, 1)
    np.testing.assert_array_almost_equal(vector, pose.as_vector())


def test_pose_map_round_trip() -> None:
    """Test that as_vector and from_vector are inverses of each other."""
    # Arrange
    pose = SE3(xyz=np.array([1.0, 2.0, 3.0]), roll_pitch_yaw=np.array([0.1, 0.2, 0.3]))
    features = [Feature(id=0, x=4.0, y=5.0), Feature(id=1, x=6.0, y=7.0)]
    pose_map = PoseMap(pose=pose, map=Map(features=features))

    # Act
    vector = pose_map.as_vector()
    round_tripped = PoseMap()
    round_tripped.from_vector(vector)

    # Assert
    np.testing.assert_array_almost_equal(round_tripped.as_vector(), vector)
    assert round_tripped.num_features == 2


def test_pose_map_num_features() -> None:
    """Test that num_features reflects the number of tracked map features."""
    # Arrange
    pose_map = PoseMap(map=Map(features=[Feature(id=0, x=1.0, y=1.0)]))

    # Act / Assert
    assert pose_map.num_features == 1
