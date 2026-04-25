#!/usr/bin/env python3

from lib.signal_schema import validate_unique_angles

CONTROLLER_ANGLES_DEG = validate_unique_angles([
    -60,
    -50,
    -40,
    -30,
    -20,
    -10,
    -5,
    0,
    5,
    10,
    20,
    30,
    40,
    50,
    60,
])

CONTROLLER_OUTPUT_WIDTH = len(CONTROLLER_ANGLES_DEG)
