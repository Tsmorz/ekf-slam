"""Basic docstring for my module."""

import numpy as np

# plot definitions
FIG_SIZE = (12, 7)
LEGEND_LOC = "upper right"
DEFAULT_NUM_STEPS = 100
PLOT_ALPHA = 0.8
PLOT_MARKER_SIZE = 3

# Kalman filter definitions
DEFAULT_VARIANCE = 1e-2
PROCESS_NOISE = 1e-2
MEASUREMENT_NOISE = 5.0
DEFAULT_CONTROL = np.array([[0.0], [0.0]])

# State space definitions
DEFAULT_DT = 1.0
DEFAULT_DISCRETIZATION = 0.05

# Logger definitions
LOG_DECIMALS = 2
LOG_LEVEL = "INFO"

# Differentiation definitions
EPSILON = 1e-3

# Map definitions
MAP_NUM_FEATURES = 100
MAP_DIM = (10, 10)
