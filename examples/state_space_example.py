"""Basic docstring for my module."""

import numpy as np

from ekf_slam_3d.modules.state_space import StateSpaceLinear


def pipeline() -> None:
    """Run the main program with this function."""
    mass = 0.5
    spring_const = 20.0
    damping = 0.4
    dt = 0.05

    mass_spring_damper_model = StateSpaceLinear(
        A=np.array([[0.0, 1.0], [-spring_const / mass, -damping / mass]]),
        B=np.array([[0.0], [1.0 / mass]]),
    )
    mass_spring_damper_model.continuous_to_discrete(dt)

    mass_spring_damper_model.step_response(delta_t=dt, plot_response=True)
    mass_spring_damper_model.impulse_response(delta_t=dt, plot_response=True)


if __name__ == "__main__":
    pipeline()
