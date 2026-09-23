"""Basic docstring for my module."""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger
from scipy.signal import cont2discrete

from config.definitions import EPSILON
from ekf_slam_3d.data_classes.state_history import StateHistory, plot_history


class StateSpaceLinear:
    """A discrete-time state-space model representation."""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray | None = None,
        C: np.ndarray | None = None,
        D: np.ndarray | None = None,
    ):
        """Initialize the state-space model.

        :param A: State transition matrix
        :param B: Control input matrix
        :param C: Observation matrix
        :param D: Direct transmission matrix
        """
        if B is None:
            B = np.zeros((A.shape[0], 1))
        if A.shape[0] != B.shape[0]:
            msg = (
                f"A and B matrices must have the same number of rows. "
                f"{A.shape[0]} != {B.shape[0]}"
            )
            logger.error(msg)
            raise ValueError(msg)
        self.A = A
        self.B = B

        if C is None:
            C = np.eye(A.shape[0])
        self.C = C

        if D is None:
            D = np.zeros((C.shape[0], B.shape[1]))
        self.D = D

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Step the state-space model by one step.

        :param x: Current state
        :param u: Control input
        :return: Next state
        """
        return self.A @ x + self.B @ u

    def step_response(self, delta_t: float, plot_response: bool = True) -> StateHistory:
        """Compute the step response of the state-space model.

        :param delta_t: Time step size
        :param plot_response: Whether to plot the step response
        :return: State history object
        """
        # Generate control input over 10 steps
        num_states = self.A.shape[0]
        num_inputs = self.B.shape[1]
        time = np.arange(0, 10, delta_t).tolist()

        control_input = len(time) * [np.ones((num_inputs, 1))]

        state_history = self.solve_trajectory(
            time=time,
            x0=np.zeros((num_states, 1)),
            control_input=control_input,
        )

        if plot_response:  # pragma: no cover
            plot_history(history=state_history, title="Step Response")

        return state_history

    def impulse_response(
        self, delta_t: float, plot_response: bool = True
    ) -> StateHistory:
        """Compute the step response of the state-space model.

        :param delta_t: Time step size
        :param plot_response: Whether to plot the step response
        :return: State history object
        """
        # Generate control input over 10 steps
        num_states = self.A.shape[0]
        num_inputs = self.B.shape[1]
        time = np.arange(0, 10, delta_t).tolist()

        control_input = len(time) * [np.zeros((num_inputs, 1))]
        control_input[0] = np.ones((num_inputs, 1))

        state_history = self.solve_trajectory(
            time=time,
            x0=np.zeros((num_states, 1)),
            control_input=control_input,
        )

        if plot_response:  # pragma: no cover
            plot_history(history=state_history, title="Impulse Response")

        return state_history

    def solve_trajectory(
        self, time: list[float], x0: np.ndarray, control_input: list[np.ndarray]
    ) -> StateHistory:
        """Solve the state-space model for given time, control input, and initial state.

        :param time: Time array
        :param x0: Initial state
        :param control_input: Control input array
        """
        state_history = StateHistory(control=control_input, time=time, state=[x0])
        x = x0
        for ii, _t in enumerate(time[:-1]):
            u = control_input[ii]
            x = self.step(x=x, u=u)
            state_history.state.append(x)
        return state_history

    def continuous_to_discrete(self, discretization_dt) -> None:
        """Convert a continuous state space to discrete state space."""
        system = (self.A, self.B, self.C, self.D)
        system_disc = cont2discrete(system, dt=discretization_dt, method="zoh")

        self.A = system_disc[0]
        self.B = system_disc[1]
        self.C = system_disc[2]
        self.D = system_disc[3]


class StateSpaceNonlinear:
    """A class for representing a nonlinear state-space model."""

    def __init__(self, motion_model: Callable):
        """Initialize a nonlinear state space model."""
        self.motion_model = motion_model

    def linearize(
        self,
        model: Callable,
        x: np.ndarray,
        u: np.ndarray,
        other_args: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Linearize a list of callables.

        :param model: List of callables representing the model to linearize
        :param x: Current state
        :param u: Control input
        :param other_args: Additional arguments passed through to `model`
        :return: Jacobian matrix
        """
        xu = np.vstack((x, u))

        if other_args is None:
            output = model(xu)
        else:
            output = model(xu, other_args)
        output_dim = output.shape[0]

        # linearize the motion model around the current state
        jacobian = np.zeros((output_dim, len(xu)))
        for xu_idx in range(len(xu)):
            result = self._derivative(
                fun=model, x=xu, x_idx=xu_idx, other_args=other_args
            )
            jacobian[:, xu_idx] = np.reshape(result, (np.shape(result)[0],))
        jacobian_x = jacobian[:, : len(x)]
        jacobian_u = jacobian[:, len(x) :]

        return jacobian_x, jacobian_u

    @staticmethod
    def _derivative(
        fun: Callable,
        x: np.ndarray,
        x_idx: int,
        other_args: Any | None = None,
    ) -> np.ndarray | float:
        """Compute the derivative of the given function.

        :param fun: Function to compute derivative
        :param x: Current state
        :param x_idx: Index of the variable to differentiate
        :param other_args: Additional arguments (e.g., list of features)
        :return: Derivative value
        """
        x_copy1, x_copy2 = copy.copy(x), copy.copy(x)
        x_copy1[x_idx, 0] = x_copy1[x_idx, 0] - EPSILON
        x_copy2[x_idx, 0] = x_copy2[x_idx, 0] + EPSILON
        if other_args is None:
            f2 = fun(x_copy2)
            f1 = fun(x_copy1)
        else:
            f2 = fun(x_copy2, other_args)
            f1 = fun(x_copy1, other_args)
        return (f2 - f1) / (2 * EPSILON)

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Step the state-space model by one step.

        :param x: Current state
        :param u: Control input
        :return: Next state
        """
        xu = np.vstack((x, u))
        return self.motion_model(xu)
