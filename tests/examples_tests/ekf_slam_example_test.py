"""Accuracy, consistency, and map-discovery checks for SLAM on every synthetic scenario."""

import numpy as np
import pytest

from ekf_slam_3d.modules.math_utils import wrap_to_pi
from examples.ekf_slam_example import SlamRun, pipeline
from examples.slam_scenarios import SCENARIOS

# (final position, landmark mean, landmark max, map shape) error tolerances, in meters
DEFAULT_TOLERANCES = (1.0, 0.5, 1.0, 0.5)
TOLERANCES = {"sparse": (1.0, 2.0, 3.0, 1.0)}

# a real loop-closure correction moves yaw by a few degrees; an unwrapped bearing
# innovation throws it by 100+ degrees in a single step
MAX_YAW_JUMP_DEG = 30.0

# largest landmark normalized estimation error squared (2 dof); seeding landmarks
# without their cross-covariance to the pose drives this into the hundreds
MAX_LANDMARK_NEES = 50.0


@pytest.fixture(scope="module", params=sorted(SCENARIOS))
def run(request: pytest.FixtureRequest) -> SlamRun:
    """Run SLAM once per scenario, deterministically seeded."""
    np.random.seed(0)
    return pipeline(show_plot=False, scenario=SCENARIOS[request.param])


def _tolerances(run: SlamRun) -> tuple[float, float, float, float]:
    return TOLERANCES.get(run.scenario.name, DEFAULT_TOLERANCES)


def _landmark_ids(run: SlamRun) -> list[int]:
    return [feature.id for feature in run.true_map.features]


def _map_shape_error(run: SlamRun) -> float:
    """Return the worst landmark error left after the best rigid alignment to the truth.

    The global frame of a range/bearing-only map is only as good as the pose it was
    anchored from, so this isolates errors in the map's shape (e.g. indexing or
    handedness bugs) from a rigid offset of the whole map.
    """
    ids = _landmark_ids(run)
    true = np.array(
        [[run.true_map.features[i].x, run.true_map.features[i].y] for i in ids]
    )
    est = np.array([run.landmark_estimate(i) for i in ids])
    true_c, est_c = true - true.mean(axis=0), est - est.mean(axis=0)
    u, _, vt = np.linalg.svd(true_c.T @ est_c)
    rotation = vt.T @ np.diag([1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    return float(np.linalg.norm(est_c - true_c @ rotation.T, axis=1).max())


def test_map_is_discovered_as_the_robot_drives(run: SlamRun) -> None:
    """Test that landmarks come into the map progressively, and all of them eventually."""
    total = len(run.true_map.features)
    assert 1 <= run.known_counts[0] < total
    assert np.all(np.diff(run.known_counts) >= 0)
    assert run.known_counts[-1] == total


def test_final_pose_and_map_accuracy(run: SlamRun) -> None:
    """Test the final pose and landmark estimates against ground truth."""
    position_tol, mean_tol, max_tol, _ = _tolerances(run)
    position_error = np.hypot(
        run.ekf.x[0, 0] - run.true_pose.x, run.ekf.x[1, 0] - run.true_pose.y
    )
    landmark_errors = [run.landmark_error(i) for i in _landmark_ids(run)]

    assert position_error < position_tol
    assert np.mean(landmark_errors) < mean_tol
    assert np.max(landmark_errors) < max_tol


def test_map_shape_accuracy(run: SlamRun) -> None:
    """Test the map's shape once any rigid offset of the whole map is removed."""
    assert _map_shape_error(run) < _tolerances(run)[3]


def test_no_sudden_orientation_jumps(run: SlamRun) -> None:
    """Test that the yaw estimate never changes by much more than the true yaw does."""
    estimated_yaw = np.array([pose[5] for pose in run.estimated_poses])
    true_yaw = np.array([pose.yaw for pose in run.true_poses])
    yaw_jump = wrap_to_pi(np.diff(estimated_yaw) - np.diff(true_yaw))

    assert np.degrees(np.max(np.abs(yaw_jump))) < MAX_YAW_JUMP_DEG


def test_filter_is_not_overconfident_about_landmarks(run: SlamRun) -> None:
    """Test that every landmark's error is plausible given the filter's own covariance."""
    for i in _landmark_ids(run):
        feature = run.true_map.features[i]
        error = run.landmark_estimate(i) - np.array([feature.x, feature.y])
        nees = error @ np.linalg.solve(run.landmark_covariance(i), error)
        assert nees < MAX_LANDMARK_NEES, f"landmark {i}"


def test_map_knowledge_improves_along_with_the_pose(run: SlamRun) -> None:
    """Test that re-observing landmarks keeps refining them after they're first seeded.

    Every landmark should end up more certain than when it was first sighted, and on
    average closer to the truth - a landmark frozen at its first (drifted) estimate
    would fail both.
    """
    ids = _landmark_ids(run)
    for i in ids:
        _, first_trace = run.first_seen[i]
        assert np.trace(run.landmark_covariance(i)) < first_trace, f"landmark {i}"

    first_errors = [run.first_seen[i][0] for i in ids]
    final_errors = [run.landmark_error(i) for i in ids]
    assert np.mean(final_errors) < np.mean(first_errors)


def test_results_do_not_depend_on_where_the_world_is() -> None:
    """Test that translating the whole world leaves every error unchanged."""
    errors = {}
    for name in ["baseline", "offset_map"]:
        np.random.seed(0)
        run = pipeline(show_plot=False, scenario=SCENARIOS[name])
        errors[name] = [run.landmark_error(i) for i in _landmark_ids(run)]

    np.testing.assert_allclose(errors["baseline"], errors["offset_map"], atol=1e-6)
