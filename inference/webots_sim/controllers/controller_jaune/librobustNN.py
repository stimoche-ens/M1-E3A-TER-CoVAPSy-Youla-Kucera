import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

K0_PATH = os.path.join(BASE_DIR, "K0_py.mat")
F_PATH = os.path.join(BASE_DIR, "HIK0_py.mat")

MODEL_PATH = os.path.join(BASE_DIR, "diffusion_uq_to_yq_model.pth")
X_SCALER_PATH = os.path.join(BASE_DIR, "x_scaler.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "y_scaler.pkl")

class robustNN():
    def __init__(self):
        pass

    def control(self, vitesse_m_s, angle_degre, tableau_lidar_mm):
        return vitesse_m_s, angle_degre