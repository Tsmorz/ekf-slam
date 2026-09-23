"""Accuracy, consistency, and map-discovery checks for SLAM on every synthetic scenario."""

import numpy as np
import pytest

from ekf_slam_3d.modules.math_utils import wrap_to_pi
from examples.ekf_slam_example import SlamRun, pipeline
from examples.slam_scenarios import SCENARIOS

# (final position, landmark mean, landmark max, map shape) error tolerances, in meters
DEFAULT_TOLERANCES = (1.0, 0.5, 1.0, 0.5)
TOLERANCES = {"sparse": (1.0, 2.0, 3.0, 1.0)}

# a real loop-closure correction moves yaw (or pitch) by a few degrees; an unwrapped
# bearing innovation throws yaw by 100+ degrees in a single step
MAX_YAW_JUMP_DEG = 30.0
MAX_PITCH_JUMP_DEG = 15.0

# largest landmark normalized estimation error squared (2 or 3 dof); seeding landmarks
# without their cross-covariance to the pose drives this into the hundreds
MAX_LANDMARK_NEES = 50.0

# worst altitude error at any step of a 3D run, in meters
MAX_ALTITUDE_ERROR = 0.25


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
    true = np.array([run.landmark_truth(i) for i in ids])
    est = np.array([run.landmark_estimate(i) for i in ids])
    true_c, est_c = true - true.mean(axis=0), est - est.mean(axis=0)
    u, _, vt = np.linalg.svd(true_c.T @ est_c)
    no_reflection = np.ones(run.landmark_dim)
    no_reflection[-1] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag(no_reflection) @ u.T
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
    true_position = [run.true_pose.x, run.true_pose.y, run.true_pose.z]
    position_error = np.linalg.norm(run.ekf.x[0:3, 0] - true_position)
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


def test_no_sudden_pitch_jumps(run: SlamRun) -> None:
    """Test that the pitch estimate never changes by much more than the true pitch does."""
    estimated_pitch = np.array([pose[4] for pose in run.estimated_poses])
    true_pitch = np.array([pose.pitch for pose in run.true_poses])
    pitch_jump = np.diff(estimated_pitch) - np.diff(true_pitch)

    assert np.degrees(np.max(np.abs(pitch_jump))) < MAX_PITCH_JUMP_DEG


def test_filter_is_not_overconfident_about_landmarks(run: SlamRun) -> None:
    """Test that every landmark's error is plausible given the filter's own covariance."""
    for i in _landmark_ids(run):
        error = run.landmark_estimate(i) - run.landmark_truth(i)
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


def test_3d_scenarios_estimate_altitude(run: SlamRun) -> None:
    """Test that 3D runs really fly at varying altitude and track it closely."""
    if not run.scenario.three_d:
        pytest.skip("planar scenario")
    true_altitude = np.array([pose.z for pose in run.true_poses])
    estimated_altitude = np.array([pose[2] for pose in run.estimated_poses])
    landmark_heights = [feature.z for feature in run.true_map.features]

    assert np.ptp(true_altitude) > run.scenario.altitude_swing
    assert np.ptp(landmark_heights) > 1.0
    assert np.max(np.abs(estimated_altitude - true_altitude)) < MAX_ALTITUDE_ERROR


def test_results_do_not_depend_on_where_the_world_is() -> None:
    """Test that translating the whole world leaves every error unchanged."""
    errors = {}
    for name in ["baseline", "offset_map"]:
        np.random.seed(0)
        run = pipeline(show_plot=False, scenario=SCENARIOS[name])
        errors[name] = [run.landmark_error(i) for i in _landmark_ids(run)]

    np.testing.assert_allclose(errors["baseline"], errors["offset_map"], atol=1e-6)
