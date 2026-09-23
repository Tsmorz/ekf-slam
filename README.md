# ekf-slam-3d

[![CI](https://github.com/tsmorz/ekf-slam/actions/workflows/ci.yml/badge.svg)](https://github.com/tsmorz/ekf-slam/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/ekf-slam-3d.svg)](https://pypi.org/project/ekf-slam-3d/)
[![Python versions](https://img.shields.io/pypi/pyversions/ekf-slam-3d.svg)](https://pypi.org/project/ekf-slam-3d/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An extended Kalman filter (EKF) library for 3D localization and SLAM (simultaneous
localization and mapping), built on numpy/scipy. It ships three estimation pipelines
that share the same filter and sensor-model code:

| Pipeline    | Pose      | Map       | What it estimates                                 |
|-------------|-----------|-----------|----------------------------------------------------|
| Localization| unknown   | known     | robot pose, from range/azimuth/elevation sightings of known landmarks |
| Mapping     | known     | unknown   | landmark positions, from sightings taken along a known trajectory |
| SLAM        | unknown   | unknown   | pose **and** map together, from sightings alone     |

Every pipeline is checked against seeded, synthetic datasets in the test suite
(`task test`), so the numbers below aren't just eyeballed off a plot - they're asserted
in CI on every change:

| Pipeline    | Final position error | Landmark position error (mean / max) |
|-------------|----------------------:|--------------------------------------:|
| Localization| < 1.0 m               | n/a (map is known)                    |
| Mapping     | n/a (pose is known)   | < 0.05 m / < 0.1 m                    |
| SLAM        | < 1.0 m               | < 0.5 m / < 1.0 m (sparse: < 2 m / < 3 m) |

(these are the tolerances enforced in `tests/examples_tests/`)

### SLAM datasets

In the SLAM pipeline the robot only sees landmarks within `SENSOR_RANGE` (10 m), so the
map is discovered as it drives, and it steers around its loop using its own pose
estimate. Each dataset in `examples/slam_scenarios.py` is built to catch a different kind
of bug:

| Scenario     | What it stresses | Known at start | Final pose error | Mean landmark error (first sighting -> end) |
|--------------|------------------|---------------:|-----------------:|--------------------------------------------:|
| `baseline`   | counterclockwise loops in a ring of 12 landmarks | 3 / 12 | 0.02 m | 0.62 -> 0.24 m |
| `clockwise`  | turning/bearing handedness | 3 / 12 | 0.03 m | 0.26 -> 0.16 m |
| `branch_cut` | yaw and bearings on the +-pi atan2 branch cut | 2 / 12 | 0.01 m | 0.21 -> 0.14 m |
| `long_run`   | 10 loops, so yaw grows to dozens of radians | 3 / 12 | 0.32 m | 0.62 -> 0.16 m |
| `sparse`     | stretches with nothing in view, then loop closure | 2 / 6 | 0.02 m | 1.31 -> 0.89 m |
| `dense`      | 24 landmarks, many in view at once | 5 / 24 | 0.01 m | 0.14 -> 0.02 m |
| `offset_map` | the whole world far from the origin | 3 / 12 | 0.02 m | 0.62 -> 0.24 m |
| `baseline_3d` | 3D: landmarks 0-8 m high, robot climbing/descending 2-6 m every loop | 3 / 12 | 0.06 m | 0.80 -> 0.29 m |
| `steep_3d` | 3D: tall map (0-12 m), 1-9 m altitude swing, large elevation angles | 4 / 12 | 0.08 m | 0.13 -> 0.02 m |
| `branch_cut_3d` | 3D while facing -x, on the +-pi branch cut | 2 / 12 | 0.03 m | 0.72 -> 0.22 m |

The `_3d` datasets run full 3D SLAM: the robot commands velocity, yaw rate, and pitch
rate (climbing or descending along its pitch), senses range, azimuth, and elevation, and
the EKF estimates its x/y/z pose along with x/y/z landmark positions. Elevation is
measured above the horizontal plane. Final pose error includes altitude, and the 3D runs
also check that altitude is tracked to within 0.25 m at every step.

For every dataset the tests check that the map is discovered progressively, that the
pose and map stay accurate, that the map's shape is right, and that yaw and pitch never
jump between steps. They also check that the filter is never overconfident about a landmark,
and that every landmark keeps improving after it's first seen. Moving the whole world
must not change any error.

## Live demo screenshots

### EKF localization (pose only, map known)
![ekf-localization](https://github.com/user-attachments/assets/c441e3ec-4151-473b-be20-dd3ba98de8b4)

### Controls / LQR steering
![controls-example](https://github.com/user-attachments/assets/f2abb831-2cf8-4599-95b8-127963a9e981)

> **Note:** These screenshots are from an earlier matplotlib-based version. The live demos now use pyqtgraph. Run `task demo:slam`, `task demo:mapping`, or `task demo:localization` to see the current plots. Screenshots will be updated in a future release.

## Install

```bash
pip install ekf-slam-3d
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add ekf-slam-3d
```

## Quickstart

```python
import numpy as np

from ekf_slam_3d.data_classes.lie_algebra import SE3
from ekf_slam_3d.data_classes.sensors import measure_gps, step_dynamics
from ekf_slam_3d.modules.kalman_extended import ExtendedKalmanFilter
from ekf_slam_3d.modules.state_space import StateSpaceNonlinear

ekf = ExtendedKalmanFilter(
    state_space_nonlinear=StateSpaceNonlinear(motion_model=step_dynamics),
    initial_x=SE3().as_vector(),
    initial_covariance=np.eye(6),
    process_noise=0.01 * np.eye(2),
    measurement_noise=0.05,
)

control = np.array([[1.0], [0.1]])  # [velocity, yaw rate]
ekf.predict(u=control)

gps_reading = np.array([[1.0], [0.05], [0.0]])
ekf.update(z=gps_reading, sensor=measure_gps, u=control, measurement_args=[])

print(ekf.x.T)  # [x, y, z, roll, pitch, yaw]
```

`ExtendedKalmanFilter` is the general-purpose filter: hand it any nonlinear motion
model and any measurement function with the `(state, args, noise=None)` signature, and
it numerically linearizes both on every `predict()`/`update()` call. The mapping and
SLAM pipelines are built from the exact same filter, just with different state layouts
and measurement functions - see `ekf_slam_3d/data_classes/sensors.py`.

`update()` optionally takes a `MeasurementSpec` with the measurement's covariance, and
a mask of which entries are angles (so bearings are compared modulo 2*pi).
`initialize_from_measurement()` adds a new landmark to the state from its first
sighting. It keeps the landmark's correlation with the pose it was seen from, so later
corrections to the pose also correct the map.

## Run the live demos

The example pipelines (with live plots) aren't part of the installed package, so clone
the repo to run them:

```bash
git clone https://github.com/tsmorz/ekf-slam.git
cd ekf-slam
task init
```

Then run any pipeline directly, or via `__main__.py -p {KF,EKF,MAPPING,SLAM,STATE_SPACE}`:

```bash
uv run python examples/ekf_localization_example.py  # pose only, map known
uv run python examples/ekf_mapping_example.py        # map only, pose known
uv run python examples/ekf_slam_example.py           # pose + map, both unknown
```

or with the demo tasks:

```bash
task demo:localization
task demo:mapping
task demo:slam                           # also just `task demo`
task demo:slam -- --scenario clockwise   # any dataset from the table above
task demo:slam3d                         # 3D SLAM (same as --scenario baseline_3d)
```

Each of these opens a live pyqtgraph window, so they need a real display - run them
locally rather than in a headless environment. The view grows to fit the robot and
whatever is known about the map, then stays put. Estimated landmarks (green `+`) are
drawn with their 2-sigma uncertainty ellipses, which shrink as they're re-observed.
The plot is top-down, so 3D runs show x/y; altitude is in the logged estimates.

## Development

```bash
task init    # uv sync
task demo    # run the SLAM demo (see also demo:slam, demo:mapping, demo:localization)
task format  # ruff format, ruff check --fix, mypy
task test    # pytest with coverage
task clean   # remove .venv, caches, build/dist artifacts
```
