"""Basic docstring for my module."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from config.definitions import (
    COVARIANCE_ELLIPSE_SIGMA,
    DEFAULT_DISCRETIZATION,
    DELTA_T,
    PAUSE_TIME,
    PLOT_ALPHA,
    VIEW_MARGIN_FRACTION,
    VIEW_MIN_MARGIN,
)
from ekf_slam_3d.data_classes.lie_algebra import SE3
from ekf_slam_3d.data_classes.slam import Map
from ekf_slam_3d.modules.controller import get_angular_velocities_for_box
from ekf_slam_3d.modules.state_space import StateSpaceLinear, StateSpaceNonlinear

if TYPE_CHECKING:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets

# (x_min, x_max, y_min, y_max)
ViewBounds = tuple[float, float, float, float]


@dataclass(frozen=True, eq=False)
class LandmarkEstimate:
    """An estimated landmark position and its 2x2 covariance, for the live plot."""

    x: float
    y: float
    covariance: np.ndarray


def expand_view_bounds(
    bounds: ViewBounds | None, xs: list[float], ys: list[float]
) -> ViewBounds | None:
    """Return view bounds grown just enough to contain every (x, y), with a margin.

    Only the sides a point falls outside of move, and only outward, so the view never
    pans or shrinks - it settles once everything known fits.

    :param bounds: current bounds, or None if no view has been set yet
    :param xs: x coordinates that must be visible
    :param ys: y coordinates that must be visible
    :return: the new bounds, or None if `bounds` already contains every point
    """
    lo_x, hi_x, lo_y, hi_y = min(xs), max(xs), min(ys), max(ys)
    if bounds is not None:
        if (
            bounds[0] <= lo_x
            and hi_x <= bounds[1]
            and bounds[2] <= lo_y
            and hi_y <= bounds[3]
        ):
            return None
        span = max(
            max(hi_x, bounds[1]) - min(lo_x, bounds[0]),
            max(hi_y, bounds[3]) - min(lo_y, bounds[2]),
        )
    else:
        span = max(hi_x - lo_x, hi_y - lo_y)
        bounds = (np.inf, -np.inf, np.inf, -np.inf)

    margin = max(VIEW_MARGIN_FRACTION * span, VIEW_MIN_MARGIN)
    return (
        min(bounds[0], lo_x - margin),
        max(bounds[1], hi_x + margin),
        min(bounds[2], lo_y - margin),
        max(bounds[3], hi_y + margin),
    )


def ellipse_geometry(covariance: np.ndarray) -> tuple[float, float, float]:
    """Return (width, height, rotation in degrees) of a 2x2 covariance's sigma ellipse.

    :param covariance: 2x2 position covariance
    :return: full axis lengths at COVARIANCE_ELLIPSE_SIGMA, and the angle of the width
        axis (the first eigenvector) from +x
    """
    eigvals, eigvecs = np.linalg.eigh(covariance)
    axes = 2 * COVARIANCE_ELLIPSE_SIGMA * np.sqrt(np.clip(eigvals, 0.0, None))
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    return float(axes[0]), float(axes[1]), angle


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
        map_known: bool = True,
    ) -> None:
        """Initialize the Kalman Filter.

        :param state_space_nl: linear state space model
        :param process_noise: Process noise covariance
        :param initial_pose: Initial state estimate
        :param sim_map: Optional map for visualization
        :param map_known: whether the map is known up front (localization) - if not
            (mapping/SLAM), the view only grows to fit landmarks as they're estimated
        :return: None
        """
        self.map_known = map_known
        self._view_bounds: ViewBounds | None = None
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
            # otherwise pyqtgraph re-fits the view to whichever transient items (rays,
            # trail, ellipses) are on screen each frame, so the view drifts every step
            plot_widget.getViewBox().disableAutoRange()
            self._sim_plot = plot_widget

            known = self.map.features if self.map_known else []
            self._include_in_view(
                xs=[self.pose.x] + [f.x for f in known],
                ys=[self.pose.y] + [f.y for f in known],
            )
        return self._sim_plot

    def _include_in_view(self, xs: list[float], ys: list[float]) -> None:
        """Grow the plot's view (never pan or shrink it) so it contains every point."""
        bounds = expand_view_bounds(self._view_bounds, xs, ys)
        if bounds is not None:
            self._view_bounds = bounds
            self.sim_plot.setRange(
                xRange=(bounds[0], bounds[1]), yRange=(bounds[2], bounds[3]), padding=0
            )

    def step(self, u: np.ndarray) -> SE3:
        """Predict the next state and error covariance.

        :param u: Control input
        """
        scale = np.reshape(np.sqrt(np.diag(self.Q)), (self.Q.shape[0], 1))
        noise = np.random.normal(loc=0.0, scale=scale, size=(self.Q.shape[0], 1))
        x = self.state_space_nl.step(x=self.pose.as_vector(), u=u + noise)
        self.pose = SE3(xyz=x[0:3], roll_pitch_yaw=x[3:6])
        return self.pose

    def append_result(
        self,
        estimate: tuple[SE3, np.ndarray],
        measurement: np.ndarray,
        show_plot: bool = True,
        landmarks: list[LandmarkEstimate] | None = None,
        measurement_stride: int = 3,
    ) -> None:
        """Update the state estimate based on an estimated pose.

        :param estimate: (estimated pose, pose covariance)
        :param measurement: the latest raw sensor measurement, used to draw rays
        :param show_plot: whether to draw this step
        :param landmarks: optional estimated landmarks (mapping/SLAM), drawn with their
            uncertainty ellipses
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

            landmarks = landmarks or []
            self._include_in_view(
                xs=[pose.x, self.pose.x] + [lm.x for lm in landmarks],
                ys=[pose.y, self.pose.y] + [lm.y for lm in landmarks],
            )
            for lm in landmarks:
                plot_items.append(
                    self.plot_ellipse(lm.x, lm.y, lm.covariance, color="g")
                )

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
            if landmarks:
                markers = self.sim_plot.plot(
                    [lm.x for lm in landmarks],
                    [lm.y for lm in landmarks],
                    pen=None,
                    symbol="+",
                    symbolBrush="g",
                    symbolSize=10,
                )
                plot_items.append(markers)
            self.last_measurement = measurement
            pg.mkQApp().processEvents()
            time.sleep(PAUSE_TIME)

            # remove the sensor measurements
            for item in plot_items:
                self.sim_plot.removeItem(item)

    def plot_covariance(
        self, pose: SE3, covariance: np.ndarray
    ) -> QtWidgets.QGraphicsEllipseItem:
        """Add a drawing of the robot position covariance to the plot."""
        return self.plot_ellipse(pose.x, pose.y, covariance[:2, :2], color="k")

    def plot_ellipse(
        self, x: float, y: float, covariance: np.ndarray, color: str
    ) -> QtWidgets.QGraphicsEllipseItem:
        """Add a COVARIANCE_ELLIPSE_SIGMA-sigma ellipse for a 2x2 position covariance."""
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtWidgets

        width, height, angle = ellipse_geometry(covariance)
        ellipse = QtWidgets.QGraphicsEllipseItem(-width / 2, -height / 2, width, height)
        ellipse.setPen(pg.mkPen(color))
        ellipse.setBrush(pg.mkBrush(None))
        ellipse.setOpacity(PLOT_ALPHA)
        ellipse.setPos(float(x), float(y))
        ellipse.setRotation(angle)
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
