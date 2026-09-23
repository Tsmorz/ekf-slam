"""Add a doc string to my files."""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from config.definitions import FIG_SIZE, LEGEND_LOC, PLOT_ALPHA, PLOT_MARKER_SIZE


@dataclass
class StateHistory:
    """A state history object to store history of state information."""

    time: list[float] = field(default_factory=list)
    state: list[np.ndarray] = field(default_factory=list)
    state_true: list[np.ndarray] = field(default_factory=list)
    control: list[np.ndarray] = field(default_factory=list)
    covariance: list[np.ndarray] = field(default_factory=list)

    def append_step(
        self,
        t: float,
        x: np.ndarray,
        x_truth: np.ndarray | None = None,
        cov: np.ndarray | None = None,
        u: np.ndarray | None = None,
    ) -> None:
        """Append state data."""
        self.time.append(t)
        self.state.append(x)
        if u is not None:
            self.control.append(u)
        if cov is not None:
            self.covariance.append(cov)
        if x_truth is not None:
            self.state_true.append(x_truth)


def plot_history(
    history: StateHistory, title: str = "State Space History"
) -> None:  # pragma: no cover
    """Plot the history of state space model.

    :param history: State history object
    :param title: Plot title
    :return: None
    """

    def _add_bounds(state, sigma, color) -> None:  # pragma: no cover
        confidence_99 = 2.576
        lower = data - confidence_99 * sigma
        upper = data + confidence_99 * sigma
        ax.fill_between(
            history.time,
            lower,
            upper,
            color=color,
            alpha=0.5,
            label=f"$x_{state} 99 C.I. $",
        )

    num_states = history.state[0].shape[0]
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=FIG_SIZE)
    plt.suptitle(title)

    for ii, ax in enumerate(axs):
        for num_state in range(num_states):
            data = np.array([arr[num_state] for arr in history.state]).flatten()

            if ii == 1:
                data /= np.amax(np.abs(data))

            p = ax.plot(
                history.time, data, "--", label=f"$x_{num_state}$", alpha=PLOT_ALPHA
            )
            if len(history.covariance) > 0:
                _add_bounds(
                    state=num_state,
                    sigma=np.array(
                        [arr[num_state, num_state] for arr in history.covariance]
                    ),
                    color=p[0].get_color(),
                )
            ax.set_ylabel("State" if ii == 0 else "Normalized State")

            # plot the ground truth if it exists
            if len(history.state_true) > 0:
                c = p[0].get_color()
                data = np.array(
                    [arr[num_state] for arr in history.state_true]
                ).flatten()
                ax.plot(
                    history.time,
                    data,
                    ".",
                    label=f"$x_{num_state} (true)$",
                    alpha=PLOT_ALPHA,
                    color=c,
                    markersize=PLOT_MARKER_SIZE,
                )
        for u in range(history.control[0].shape[1]):
            control = [arr[u] for arr in history.control]
            if ii == 1:
                control /= np.amax(np.abs(control))
            ax.step(history.time, control, label="control input", alpha=PLOT_ALPHA)

    for ax in axs:
        ax.set_xlabel("Time (s)")
        ax.grid(True)
        ax.legend(loc=LEGEND_LOC)
        ax.set_xlim(min(history.time) - 0.5, max(history.time) + 0.5)

    plt.show()
    plt.close()


@dataclass
class Velocity:
    """Represent the velocity as a dataclass."""

    x: float | np.ndarray
    y: float | np.ndarray
    z: float | np.ndarray
