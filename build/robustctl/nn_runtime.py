#!/usr/bin/env python3

from collections import deque
import importlib.util
from pathlib import Path
import sys

import numpy as np

try:
    from . import conf as robust_conf
except ImportError:
    import conf as robust_conf


def _load_nn_conf():
    root = robust_conf.PROJECT_ROOT
    nntrain_dir = root / "build" / "nntrain"
    if str(nntrain_dir) not in sys.path:
        sys.path.insert(0, str(nntrain_dir))

    # Pool modules import a bare `conf`; bind that name explicitly to
    # nntrain/conf.py so root conf packages cannot win by import-order luck.
    conf_path = nntrain_dir / "conf.py"
    loaded_conf = sys.modules.get("conf")
    loaded_path = Path(getattr(loaded_conf, "__file__", "")) if loaded_conf else None
    if loaded_path and loaded_path.resolve() == conf_path.resolve():
        return loaded_conf

    spec = importlib.util.spec_from_file_location("conf", conf_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["conf"] = module
    spec.loader.exec_module(module)
    return module


def _tensor_stats(torch, width):
    return {
        "size": torch.Size([width]),
        "train_offset": torch.zeros(width, dtype=torch.float32),
        "train_scale": torch.ones(width, dtype=torch.float32),
    }


class NeuralYoulaLSTMPredictor:
    def __init__(
        self,
        weights_path=None,
        device="cpu",
        blend=None,
        prediction_clip_abs_m=None,
        expected_output_width=None,
    ):
        self.nn_conf = _load_nn_conf()
        self.cfg = self.nn_conf.UQYQ_LSTM
        self.weights_path = Path(weights_path or self.nn_conf.MYLSTM_WEIGHTS_PATH)
        self.device_name = device
        self.expected_output_width = (
            None if expected_output_width is None else int(expected_output_width)
        )
        self.output_width = int(self.cfg["output_width"])
        self.past_feature_width = 2 + self.output_width
        self.blend = float(self.cfg["blend"] if blend is None else blend)
        self.prediction_clip_abs_m = float(
            self.cfg["prediction_clip_abs_m"]
            if prediction_clip_abs_m is None
            else prediction_clip_abs_m
        )
        self.prediction_delta_clip_abs_m = float(self.cfg["prediction_delta_clip_abs_m"])
        self.torch = None
        self.model = None
        self.output_bias = None
        self.history = deque(maxlen=int(self.cfg["past_window"]))
        self._load_model()

    @property
    def enabled(self):
        return self.model is not None

    def _load_model(self):
        if not self.weights_path.exists():
            return

        import torch
        from pool.MyLSTM import MyLSTM

        self.torch = torch
        raw_state_dict = torch.load(self.weights_path, map_location=self.device_name)
        clean_state_dict = {
            key.replace("_orig_mod.", ""): value
            for key, value in raw_state_dict.items()
        }
        inferred_width = self._infer_output_width(clean_state_dict)
        self.output_width = inferred_width or self.output_width
        self.past_feature_width = 2 + self.output_width
        if self.expected_output_width is not None and self.output_width != self.expected_output_width:
            raise ValueError(
                f"MyLSTM output width is {self.output_width}, but active controller expects "
                f"{self.expected_output_width}. Regenerate UQYQ and retrain MyLSTM."
            )

        dataset_stats = {
            self.nn_conf.CMD_SPEED: _tensor_stats(torch, 1),
            self.nn_conf.CMD_ANGLE: _tensor_stats(torch, 1),
            self.nn_conf.MES_LIDAR: _tensor_stats(torch, self.output_width),
        }
        model = MyLSTM(
            dataset_stats=dataset_stats,
            hidden_dim=int(self.cfg["hidden_dim"]),
        )
        model.load_state_dict(clean_state_dict)
        model.to(self.device_name)
        model.eval()
        self.model = model
        self.output_bias = self._predict_raw(
            np.zeros((int(self.cfg["past_window"]), self.past_feature_width), dtype=np.float32),
            np.zeros((int(self.cfg["future_window"]), 2), dtype=np.float32),
        )

    def _infer_output_width(self, state_dict):
        for key in ("output.weight", "module.output.weight"):
            if key in state_dict:
                return int(state_dict[key].shape[0])
        for key, value in state_dict.items():
            if key.endswith("output.weight") and hasattr(value, "shape") and len(value.shape) == 2:
                return int(value.shape[0])
        return None

    def observe(self, y_q, u_q):
        u_q = np.asarray(u_q, dtype=np.float32)
        if u_q.size != self.output_width:
            raise ValueError(
                f"expected u_q width {self.output_width}, got {u_q.size}. "
                "Regenerate UQYQ/retrain the NN for the active controller angles."
            )
        sample = np.concatenate([
            np.asarray(y_q, dtype=np.float32).reshape(2),
            u_q.reshape(self.output_width),
        ])
        self.history.append(sample)

    def _past_array(self):
        if not self.history:
            self.history.append(np.zeros(self.past_feature_width, dtype=np.float32))

        rows = list(self.history)
        while len(rows) < int(self.cfg["past_window"]):
            rows.insert(0, rows[0].copy())

        return np.asarray(rows[-int(self.cfg["past_window"]):], dtype=np.float32)

    def _predict_raw(self, past_array, future_array):
        past_tensor = self.torch.from_numpy(past_array).unsqueeze(0).to(self.device_name)
        future_tensor = self.torch.from_numpy(future_array).unsqueeze(0).to(self.device_name)
        with self.torch.no_grad():
            prediction = self.model(past_tensor, future_tensor)
        return prediction[0, 0].detach().cpu().numpy().astype(float)

    def predict_uq(self, future_y_q=None):
        if not self.enabled:
            return None

        future_window = int(self.cfg["future_window"])
        if future_y_q is None:
            future = np.zeros((future_window, 2), dtype=np.float32)
        else:
            future = np.asarray(future_y_q, dtype=np.float32)
            future = np.broadcast_to(future.reshape(1, 2), (future_window, 2)).copy()

        predicted = self._predict_raw(self._past_array(), future)
        if self.output_bias is not None:
            predicted = predicted - self.output_bias
        return np.clip(
            predicted,
            -self.prediction_clip_abs_m,
            self.prediction_clip_abs_m,
        )

    def blend_eps(self, current_eps, predicted_eps):
        current_eps = np.asarray(current_eps, dtype=float)
        predicted_eps = np.asarray(predicted_eps, dtype=float)
        bounded_prediction = current_eps + np.clip(
            predicted_eps - current_eps,
            -self.prediction_delta_clip_abs_m,
            self.prediction_delta_clip_abs_m,
        )
        return (1.0 - self.blend) * current_eps + self.blend * bounded_prediction
