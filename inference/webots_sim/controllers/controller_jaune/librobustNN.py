import importlib.util
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent

LOCAL_ARTIFACT_PATH = BASE_DIR / "robust_controller.json"


def _load_project_paths():
    for parent in [BASE_DIR, *BASE_DIR.parents]:
        path = parent / "conf" / "paths.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location("_ter_conf_paths", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("could not locate conf/paths.py")


_paths = _load_project_paths()
ROBUSTCTL_PARENT = _paths.BUILD_DIR

if str(ROBUSTCTL_PARENT) not in sys.path:
    sys.path.insert(0, str(ROBUSTCTL_PARENT))

from robustctl import conf as robust_conf
from robustctl.kcontroller import load_controller, synthesize_controller
from robustctl.nn_runtime import NeuralYoulaLSTMPredictor

class robustNN():
    def __init__(self, artifact_path=None, controller_name="K0", use_nn=None):
        self.controller_name = controller_name
        self.artifact_path = Path(artifact_path) if artifact_path is not None else self._default_artifact_path()
        self.controller = self._load_controller()
        self.last_y_q = None
        self.nn_predictor = self._load_nn_predictor(use_nn)

    def _default_artifact_path(self):
        if LOCAL_ARTIFACT_PATH.exists():
            return LOCAL_ARTIFACT_PATH
        return robust_conf.CONTROLLER_ARTIFACT_PATH

    def _load_controller(self):
        if self.artifact_path.exists():
            return load_controller(self.artifact_path, name=self.controller_name)

        # Webots can still run before the artifact is generated; synthesize from
        # the shared robustctl defaults, but keep build_controller.py as the
        # canonical way to persist and deploy parameters.
        return synthesize_controller(name=self.controller_name)

    def _load_nn_predictor(self, use_nn):
        if use_nn is False:
            return None
        try:
            predictor = NeuralYoulaLSTMPredictor(
                expected_output_width=len(self.controller.angles),
            )
        except Exception as exc:
            print(f"[robustNN] NN disabled: {exc}")
            return None
        if predictor.enabled:
            return predictor
        if use_nn is True:
            print("[robustNN] NN requested, but no MyLSTM weights were found.")
        return None

    def control(self, vitesse_m_s, angle_degre, tableau_lidar_mm):
        linear_command = self.controller.command_from_lidar(tableau_lidar_mm)

        if self.nn_predictor is None:
            return linear_command.speed_m_s, linear_command.steering_angle_deg

        if self.last_y_q is None:
            self.last_y_q = np.zeros(2, dtype=float)

        self.nn_predictor.observe(self.last_y_q, linear_command.eps)
        predicted_eps = self.nn_predictor.predict_uq(future_y_q=np.zeros(2, dtype=float))

        if predicted_eps is None:
            command = linear_command
        else:
            blended_eps = self.nn_predictor.blend_eps(linear_command.eps, predicted_eps)
            command = self.controller.command_from_eps(blended_eps)

        nominal = np.array([self.controller.v_nom, self.controller.delta_nom], dtype=float)
        output_delta = np.array([command.speed_m_s, command.steering_angle_deg]) - nominal
        self.last_y_q = output_delta - linear_command.correction
        return command.speed_m_s, command.steering_angle_deg
