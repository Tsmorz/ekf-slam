"""Basic docstring for my module."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from config.definitions import (
    DEFAULT_DISCRETIZATION,
    DELTA_T,
    PAUSE_TIME,
    PLOT_ALPHA,
)
from ekf_slam_3d.data_classes.lie_algebra import SE3
from ekf_slam_3d.data_classes.map import Feature
from ekf_slam_3d.data_classes.slam import Map
from ekf_slam_3d.modules.controller import get_angular_velocities_for_box
from ekf_slam_3d.modules.state_space import StateSpaceLinear, StateSpaceNonlinear

if TYPE_CHECKING:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets


def mass_spring_damper_model(
    mass: float = 0.5,
    spring_const: float = 20.0,
    damping: float = 0.4,
    discretization_dt: float = DEFAULT_DISCRETIZATION,
) -> StateSpaceLinear:  # pragma: no cover
    """Calculate a simple mass spring damper model.

    :param mass: Mass of the system
    :param spring_const: Spring constant
    :param damping: Damping coefficient
    :param discretization_dt: Desired discrete time step size
    :return: state-space model
    """
    model = StateSpaceLinear(
        A=np.array([[0.0, 1.0], [-spring_const / mass, -damping / mass]]),
        B=np.array([[0.0], [1.0 / mass]]),
    )
    model.continuous_to_discrete(discretization_dt)
    return model


class SlamSimulator:
    """Kalman filter implementation."""

    def __init__(
        self,
        state_space_nl: StateSpaceNonlinear,
        process_noise: np.ndarray,
        initial_pose: SE3,
        sim_map: Map,
    ) -> None:
        """Initialize the Kalman Filter.

        :param state_space_nl: linear state space model
        :param process_noise: Process noise covariance
        :param initial_pose: Initial state estimate
        :param sim_map: Optional map for visualization
        :return: None
        """
        self.state_space_nl = state_space_nl
        self.Q: np.ndarray = process_noise
        self.pose: SE3 = initial_pose
        self.history: list[tuple[SE3, SE3]] = []
        self.map: Map = sim_map
        self.last_measurement: np.ndarray = np.array([])
        self.time_stamps: np.ndarray = np.arange(0.0, 20000 / DELTA_T, DELTA_T)
        self.controls = get_angular_velocities_for_box(
            steps=len(self.time_stamps), radius_steps=8
        )
        # created lazily on first plotted result, so headless/test usage (show_plot=False)
        # never opens a GUI window
        self._sim_plot: pg.PlotWidget | None = None

    @property
    def sim_plot(self) -> pg.PlotWidget:
        """Return the plot widget for this simulation, creating it on first use."""
        import pyqtgraph as pg

        if self._sim_plot is None:
            pg.setConfigOption("background", "w")
            pg.setConfigOption("foreground", "k")
            pg.mkQApp("EKF SLAM Simulation")
            plot_widget = pg.PlotWidget(title="Robot Localization")
            plot_widget.setAspectLocked(True)
            plot_widget.showGrid(x=True, y=True)
            plot_widget.setLabel("bottom", "x position")
            plot_widget.setLabel("left", "y position")
            plot_widget.resize(800, 800)
            plot_widget.plot(
                [feature.x for feature in self.map.features],
                [feature.y for feature in self.map.features],
                pen=None,
                symbol="star",
                symbolBrush="k",
            )
            plot_widget.show()
            self._sim_plot = plot_widget
        return self._sim_plot

    def step(self, u: np.ndarray) -> SE3:
        """Predict the next state and error covariance.

        :param u: Control input
        """
        scale = np.reshape(np.diag(self.Q), (self.Q.shape[0], 1))
        noise = np.random.normal(loc=0.0, scale=scale, size=(self.Q.shape[0], 1))
        x = self.state_space_nl.step(x=self.pose.as_vector(), u=u + noise)
        self.pose = SE3(xyz=x[0:3], roll_pitch_yaw=x[3:6])
        return self.pose

    def append_result(
        self,
        estimate: tuple[SE3, np.ndarray],
        measurement: np.ndarray,
        show_plot: bool = True,
        estimated_features: list[Feature] | None = None,
        measurement_stride: int = 3,
    ) -> None:
        """Update the state estimate based on an estimated pose.

        :param estimate: (estimated pose, pose covariance)
        :param measurement: the latest raw sensor measurement, used to draw rays
        :param show_plot: whether to draw this step
        :param estimated_features: optional estimated landmark positions (mapping/SLAM)
        :param measurement_stride: number of values packed per feature in `measurement`
            (3 for distance/azimuth/elevation, 2 for distance/azimuth)
        """
        pose, cov = estimate
        self.history.append((pose, self.pose))

        old_poses = []
        for old_pose, _ in self.history[-20:]:
            old_poses.append(old_pose)
        plot_items: list[QtWidgets.QGraphicsItem] = []
        if show_plot:
            import pyqtgraph as pg

            plot_items.append(self.pose.plot_se3(plot=self.sim_plot, color="red"))

            for ii, old_pose in enumerate(old_poses):
                plot_items.append(
                    old_pose.plot_se3(
                        plot=self.sim_plot, color="blue", alpha=ii / len(old_poses)
                    )
                )
            plot_items.append(self.plot_covariance(pose=pose, covariance=cov))
            plot_items.extend(
                self.plot_measurement(
                    pose=pose, measurement=measurement, stride=measurement_stride
                )
            )
            if estimated_features:
                landmarks = self.sim_plot.plot(
                    [f.x for f in estimated_features],
                    [f.y for f in estimated_features],
                    pen=None,
                    symbol="+",
                    symbolBrush="g",
                    symbolSize=10,
                )
                plot_items.append(landmarks)
            self.last_measurement = measurement
            pg.mkQApp().processEvents()
            time.sleep(PAUSE_TIME)

            # remove the sensor measurements
            for item in plot_items:
                self.sim_plot.removeItem(item)

    def plot_covariance(
        self, pose: SE3, covariance: np.ndarray
    ) -> QtWidgets.QGraphicsEllipseItem:
        """Add a drawing of the robot covariance to the plot."""
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtWidgets

        xy_cov = np.linalg.eigvalsh(covariance[:2, :2])
        xy_cov = np.clip(xy_cov, a_min=-20, a_max=20)
        width, height = float(xy_cov[0]), float(xy_cov[1])

        ellipse = QtWidgets.QGraphicsEllipseItem(-width / 2, -height / 2, width, height)
        ellipse.setPen(pg.mkPen("k"))
        ellipse.setBrush(pg.mkBrush(None))
        ellipse.setOpacity(PLOT_ALPHA)
        ellipse.setPos(float(pose.x), float(pose.y))
        ellipse.setRotation(np.rad2deg(np.arctan2(xy_cov[1], xy_cov[0])))
        self.sim_plot.addItem(ellipse)
        return ellipse

    def plot_measurement(
        self,
        measurement: np.ndarray,
        pose: SE3,
        stride: int = 3,
    ) -> list[pg.PlotDataItem]:
        """Plot the simulation results."""
        rays: list[pg.PlotDataItem] = []
        if measurement.size > 0 and not np.array_equal(
            self.last_measurement, measurement
        ):
            import pyqtgraph as pg

            distance = measurement[0::stride, 0]
            azimuth = measurement[1::stride, 0]
            for dist, azi in zip(distance, azimuth, strict=False):
                x1, x2 = pose.x, pose.x + dist * np.cos(pose.yaw + azi)
                y1, y2 = pose.y, pose.y + dist * np.sin(pose.yaw + azi)
                ray = self.sim_plot.plot([x1, x2], [y1, y2], pen=pg.mkPen("k", width=1))
                ray.setOpacity(0.2)
                rays.append(ray)
            return rays
        return rays
