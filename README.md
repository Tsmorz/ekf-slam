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

Every pipeline is checked against a seeded, synthetic dataset in the test suite
(`task test`), so the numbers below aren't just eyeballed off a plot - they're asserted
in CI on every change:

| Pipeline    | Final position error | Landmark position error (mean / max) |
|-------------|----------------------:|--------------------------------------:|
| Localization| < 1.0 m               | n/a (map is known)                    |
| Mapping     | n/a (pose is known)   | < 0.05 m / < 0.1 m                    |
| SLAM        | < 1.0 m               | < 0.5 m / < 1.0 m                     |

(these are the tolerances enforced in `tests/examples_tests/`; a typical run lands well
inside them - e.g. the SLAM pipeline's seeded test run lands around 0.2 m for both pose
and landmark error)

# controls example
![controls-example](https://github.com/user-attachments/assets/f2abb831-2cf8-4599-95b8-127963a9e981)

# ekf localization
![ekf-localization](https://github.com/user-attachments/assets/c441e3ec-4151-473b-be20-dd3ba98de8b4)

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

## Development

```bash
task init    # uv sync
task format  # ruff format, ruff check --fix, mypy
task test    # pytest with coverage
task clean   # remove .venv, caches, build/dist artifacts
```
