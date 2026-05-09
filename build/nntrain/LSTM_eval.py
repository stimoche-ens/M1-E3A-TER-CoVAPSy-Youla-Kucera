#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np

_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORTS))

from conf.paths import BUILD_DIR

if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

from robustctl.nn_runtime import NeuralYoulaLSTMPredictor


def main():
    predictor = NeuralYoulaLSTMPredictor()
    if not predictor.enabled:
        raise FileNotFoundError("No active MyLSTM weights found in conf/current_artifacts.json")

    y_q = np.zeros(2, dtype=float)
    u_q = np.zeros(predictor.output_width, dtype=float)
    for _ in range(predictor.cfg["past_window"]):
        predictor.observe(y_q, u_q)

    prediction = predictor.predict_uq(future_y_q=np.zeros(2, dtype=float))
    print("Active MyLSTM weights:", predictor.weights_path)
    print("Predicted future u_q[0]:", prediction)


if __name__ == "__main__":
    main()
