#!/usr/bin/env python3

import argparse
import csv
import os
import sys

import numpy as np
from joblib import load

try:
    from ament_index_python.packages import get_package_share_directory
    # Dynamically resolve the ROS 2 share directory for the package models
    DEFAULT_MODEL_DIR = os.path.join(get_package_share_directory('YKREN'), 'models')
except Exception:
    # Fallback to the script's current directory if ROS 2 is not sourced
    DEFAULT_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer on a rosbag-converted CSV and print lidar values at -90,-60,0,60,90 degrees plus predicted commands."
    )
    parser.add_argument("csv_path", help="Path to the rosbag-converted CSV file")
    parser.add_argument(
        "--model-dir",
        #default=os.path.dirname(os.path.abspath(__file__)),
        default=DEFAULT_MODEL_DIR,
        help="Directory containing trained_model.joblib, scaler_X.joblib, scaler_Y.joblib",
    )
    parser.add_argument("--start", type=int, default=0, help="First row index to process (0-based)")
    parser.add_argument("--count", type=int, default=20, help="Number of rows to process")
    parser.add_argument(
        "--angle-min",
        type=float,
        default=-100.0,
        help="Minimum lidar angle of the csv scan in degrees",
    )
    parser.add_argument(
        "--angle-max",
        type=float,
        default=100.0,
        help="Maximum lidar angle of the csv scan in degrees",
    )
    parser.add_argument(
        "--selected-angles",
        type=float,
        nargs="+",
        default=[-90.0, -60.0, 0.0, 60.0, 90.0],
        help="Angles to print from the lidar scan",
    )
    return parser.parse_args()


def load_model_and_scalers(model_dir):
    model_path = os.path.join(model_dir, "trained_model.joblib")
    scaler_X_path = os.path.join(model_dir, "scaler_X.joblib")
    scaler_Y_path = os.path.join(model_dir, "scaler_Y.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(scaler_X_path):
        raise FileNotFoundError(f"Scaler X not found: {scaler_X_path}")
    if not os.path.exists(scaler_Y_path):
        raise FileNotFoundError(f"Scaler y not found: {scaler_Y_path}")

    model = load(model_path)
    scaler_X = load(scaler_X_path)
    scaler_Y = load(scaler_Y_path)
    return model, scaler_X, scaler_Y


def parse_ranges_string(ranges_text):
    values = []
    for token in ranges_text.split(";"):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "inf":
            values.append(np.inf)
        elif token.lower() == "-inf":
            values.append(-np.inf)
        elif token.lower() == "nan":
            values.append(np.nan)
        else:
            values.append(float(token))
    return np.array(values, dtype=np.float32)


def selected_lidar_values(scan, angles, angle_min, angle_max):
    n = len(scan)
    if n < 2:
        raise ValueError("Lidar scan must contain at least 2 values")
    step = (angle_max - angle_min) / (n - 1)
    result = {}
    for angle in angles:
        idx = int(round((angle - angle_min) / step))
        idx = max(0, min(idx, n - 1))
        result[angle] = float(scan[idx])
    return result


def main():
    args = parse_args()
    model, scaler_X, scaler_Y = load_model_and_scalers(args.model_dir)

    csv_path = os.path.abspath(args.csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    end_index = min(len(rows), args.start + args.count)
    if args.start >= len(rows):
        raise IndexError(f"Start index {args.start} is out of bounds, file has {len(rows)} rows")

    print(
        "index | elapsed_sec | steering_gt | speed_gt | " + \
        " | ".join([f"lidar_{int(angle)}" for angle in args.selected_angles]) + " | pred_steering | pred_speed"
    )

    base_timestamp = None
    for idx in range(args.start, end_index):
        row = rows[idx]
        timestamp_text = row.get("timestamp", "")
        ranges_text = row.get("ranges", "")
        steering_gt = row.get("steering", "")
        speed_gt = row.get("speed", "")

        try:
            timestamp = float(timestamp_text)
        except (ValueError, TypeError):
            timestamp = None

        if base_timestamp is None and timestamp is not None:
            base_timestamp = timestamp

        elapsed_sec = ""
        if timestamp is not None and base_timestamp is not None:
            elapsed_sec = f"{timestamp - base_timestamp:.1f}"

        scan = parse_ranges_string(ranges_text)
        scan = np.nan_to_num(scan, nan=30.0, posinf=30.0, neginf=0.0)

        selected = selected_lidar_values(scan, args.selected_angles, args.angle_min, args.angle_max)
        lidar_features = [selected[angle] for angle in args.selected_angles]

        lidar_scaled = scaler_X.transform([scan])
        pred_scaled = model.predict(lidar_scaled)
        pred = scaler_Y.inverse_transform(pred_scaled)[0]
        pred_steering = float(pred[0])
        pred_speed = float(pred[1])

        pred_steering = max(min(pred_steering, 18.0), -18.0)
        pred_speed = max(min(pred_speed, 1.0), 0.0)

        def format_value(value):
            try:
                return f"{float(value):.1f}"
            except (ValueError, TypeError):
                return str(value)

        values = [
            str(idx),
            elapsed_sec,
            format_value(steering_gt),
            format_value(speed_gt),
        ] + [f"{lidar_features[i]:.1f}" for i in range(len(lidar_features))] + [
            f"{pred_steering:.1f}",
            f"{pred_speed:.1f}",
        ]
        print(" | ".join(values))


if __name__ == "__main__":
    main()
