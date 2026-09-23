"""Basic docstring for my module."""

import argparse
import subprocess
from enum import Enum, auto

from loguru import logger


class Pipeline(Enum):
    """Create an enumerator to choose which pipeline to run."""

    KF = auto()
    EKF = auto()
    MAPPING = auto()
    SLAM = auto()
    CONTROLLER = auto()
    STATE_SPACE = auto()


if __name__ == "__main__":  # pragma: no cover
    """Run the main program with this function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pipeline",
        action="store",
        type=str,
        required=True,
        help=f"Choose which pipeline to run - {[v.name for v in Pipeline]}",
    )
    args = parser.parse_args()

    pipeline_id = args.pipeline

    if pipeline_id == Pipeline.KF.name:
        subprocess.run(["python", "examples/kf_example.py"], check=True)
    elif pipeline_id == Pipeline.EKF.name:
        subprocess.run(["python", "examples/ekf_localization_example.py"], check=False)
    elif pipeline_id == Pipeline.MAPPING.name:
        subprocess.run(["python", "examples/ekf_mapping_example.py"], check=False)
    elif pipeline_id == Pipeline.SLAM.name:
        subprocess.run(["python", "examples/ekf_slam_example.py"], check=False)
    elif pipeline_id == Pipeline.STATE_SPACE.name:
        subprocess.run(["python", "examples/state_space_example.py"], check=False)
    else:
        msg = f"Invalid pipeline number: {pipeline_id}"
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Program complete.")
