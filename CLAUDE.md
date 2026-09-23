# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An extended Kalman filter (EKF) library for 3D localization and SLAM (simultaneous localization and mapping), built on numpy/scipy. Package name is `ekf_slam_3d` (PyPI: `ekf-slam-3d`), dependency-managed with uv.

## Commands

Uses uv + a Taskfile (go-task). Python 3.13 required. Runtime deps live in `[project.dependencies]`; test/dev tooling (pytest, pytest-cov, pre-commit) is in the `dev` group under `[dependency-groups]`.

```bash
task init    # uv sync
task format  # ruff format, ruff check --fix, mypy over ekf_slam_3d/ tests/ config/ examples/
task test    # pytest with coverage over ekf_slam_3d/ (term-missing report, no-cov-on-fail)
task clean   # remove .venv, caches, build/dist artifacts
```

Run a single test:
```bash
uv run pytest tests/modules_tests/kalman_test.py::test_name
```

Run one of the example pipelines directly (also invocable via `__main__.py -p {KF,EKF,SLAM,STATE_SPACE}`):
```bash
uv run python examples/ekf_localization_example.py
```

Pre-commit hooks (flake8, mypy, ruff, ruff-format) run on commit; the `pytest` hook is skipped in CI (`.pre-commit-config.yaml`). Note: `ruff` itself is expected on `PATH` (not a project dependency) — both `task format` and the old `make format` invoke it bare via `uv run ruff`, which falls back to a `PATH` install.

## Architecture

Two-layer design: `ekf_slam_3d/data_classes/` holds state/geometry representations and measurement models; `ekf_slam_3d/modules/` holds the estimation/control algorithms that operate on them. `config/definitions.py` centralizes all tunable constants (noise sigmas, plotting, epsilon for numerical differentiation, etc.) — new constants belong there, not inline.

**State representation** (`data_classes/lie_algebra.py`): `SE3` is the pose class (x, y, z, roll, pitch, yaw), convertible to/from a 6-vector and a 4x4 homogeneous transform via `as_vector()`/`as_matrix()`. `state_to_se3()` converts the first 6 elements of any state vector into an `SE3`. All state vectors elsewhere in the codebase are numpy column vectors (shape `(n, 1)`), with pose occupying indices `0:6`.

**Map/measurement layer** (`data_classes/map.py`, `data_classes/sensors.py`): `Feature`/`Map` hold landmark positions. `sensors.py` defines measurement functions (`measure_gps`, `measure_distance`, `measure_azimuth`, `measure_elevation`, `measure_distance_azimuth_elevation`) that all share the signature `(state, features, noise=None) -> np.ndarray`, and `step_dynamics()` which is the nonlinear motion model (bicycle-style kinematics). The `Sensor` enum wires sensor names to their measurement functions so filters can be handed `Sensor.X.func` generically. `PoseMap` (`data_classes/slam.py`) packs pose + map features into a single EKF state vector for SLAM (pose in `0:6`, then interleaved feature x/y pairs).

**Estimation layer** (`modules/`):
- `state_space.py`: `StateSpaceLinear` (A/B/C/D matrices, `step`, `continuous_to_discrete` via `scipy.signal.cont2discrete`) and `StateSpaceNonlinear`, which wraps an arbitrary motion-model callable and numerically linearizes it (central-difference Jacobian, see `EPSILON` in `config/definitions.py`) to produce local A/B matrices — this is what makes the EKF "extended".
- `kalman.py` / `kalman_extended.py`: `KalmanFilter` operates on a `StateSpaceLinear`; `ExtendedKalmanFilter` operates on a `StateSpaceNonlinear`, re-linearizing the motion model on `predict()` and both the motion model and the given sensor function on `update()` (the sensor function itself is passed into `update()`, enabling multi-sensor fusion with a single filter instance). Both keep covariance symmetric via `math_utils.symmetrize_matrix()` after every step.
- `controller.py`: LQR / pole-placement control (`LQRController`, `full_state_feedback`), independent of the filtering code — used to drive simulated trajectories (e.g. `get_angular_velocities_for_box` for box-path trajectories).
- `simulators.py`: `SlamSimulator` drives ground-truth motion, live matplotlib visualization (pose, covariance ellipse, measurement rays), and feeds measurements to a filter in the example pipelines.
- `math_utils.py`: standalone linear-algebra/rotation helpers (`skew_matrix`, `matrix_exponential` via eigendecomposition or Jordan form for non-diagonalizable matrices, `align_to_gravity`, IMU-style dead-reckoning helpers).

**Entry point** (`__main__.py`): a `Pipeline` enum (`KF`, `EKF`, `SLAM`, `CONTROLLER`, `STATE_SPACE`) dispatches to scripts under `examples/` via subprocess; `SLAM` is not yet implemented (see `Map`/`PoseMap` plumbing that anticipates it).

Tests in `tests/` mirror the `ekf_slam_3d/` package layout (`data_classes_tests/`, `modules_tests/`).
