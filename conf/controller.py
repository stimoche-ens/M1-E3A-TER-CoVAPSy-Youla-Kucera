#!/usr/bin/env python3

from lib.signal_schema import validate_unique_angles

CONTROLLER_ANGLES_DEG = validate_unique_angles([
    -60,
    -30,
    0,
    30,
    60,
])

CONTROLLER_OUTPUT_WIDTH = len(CONTROLLER_ANGLES_DEG)
