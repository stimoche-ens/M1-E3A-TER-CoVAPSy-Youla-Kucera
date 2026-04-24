#!/usr/bin/env python3

import argparse
import csv
import math
import os
import sys
import yaml

import numpy as np

# ROS 2 imports required for reading bags and deserializing messages
try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as e:
    print(f"ROS 2 Python libraries not found. Ensure you have sourced your ROS 2 environment. Error: {e}", file=sys.stderr)
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a ROS 2 rosbag (.mcap or directory) into a synchronized CSV file."
    )
    parser.add_argument(
        "bag_paths", 
        nargs="+",
        help="Paths to the rosbag directories or specific .mcap files."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-o", "--output", 
        default="output.csv", 
        help="Path to the output CSV file (concatenates all rosbags). Mutually exclusive with -O."
    )
    group.add_argument(
        "-O", "--output-dir", 
        default=None, 
        help="Directory to output individual CSV files per rosbag. Mutually exclusive with -o."
    )
    parser.add_argument(
        "-r", "--recursive", 
        action="store_true", 
        help="Recursively search for rosbags within the provided directories."
    )
    parser.add_argument(
        "--trigger-topics", 
        nargs="+",
        default=["/scan"],
        help="The topic(s) that trigger a new row to be written to the CSV (default: /scan). Multiple topics can be provided."
    )
    parser.add_argument(
        "--keep-cols", 
        nargs="+", 
        default=None, 
        help="Optional list of specific column names to keep. If omitted, all extracted columns are kept."
    )
    return parser.parse_args()


def get_reader(bag_path):
    """Initializes and returns the rosbag2 reader based on the input path type."""
    reader = rosbag2_py.SequentialReader()
    
    # Determine storage ID based on whether the path is a directory or a direct .mcap file
    if os.path.isfile(bag_path) and bag_path.endswith('.mcap'):
        storage_id = 'mcap'
    else:
        storage_id = '' # Leave empty to let ROS 2 auto-detect from metadata.yaml

    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id=storage_id
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    reader.open(storage_options, converter_options)
    return reader


def extract_topic_types(reader):
    """Creates a mapping of topic names to their corresponding message classes."""
    topic_types = reader.get_all_topics_and_types()
    type_map = {}
    for topic_meta in topic_types:
        # topic_meta.type is a string like 'sensor_msgs/msg/LaserScan'
        try:
            msg_class = get_message(topic_meta.type)
            type_map[topic_meta.name] = msg_class
        except Exception as e:
            print(f"Warning: Could not load message class for {topic_meta.type}. Skipping topic {topic_meta.name}.", file=sys.stderr)
    return type_map


def flatten_message(topic_name, msg):
    """
    Parses specific ROS 2 messages and flattens them into a dictionary of scalar values.
    Extend this function to handle custom message types as needed.
    """
    flat_data = {}
    
    # Process LaserScan
    if hasattr(msg, 'ranges') and hasattr(msg, 'angle_increment'):
        for i, r in enumerate(msg.ranges):
            # Calculate angle in radians, then convert to degrees
            angle_rad = msg.angle_min + i * msg.angle_increment
            angle_deg = int(round(math.degrees(angle_rad)))
            
            # Clean up nan/inf values for CSV
            if math.isinf(r) or math.isnan(r):
                val = np.nan
            else:
                val = float(r)
                
            flat_data[f"lidar[{angle_deg}]"] = val

    # Process AckermannDrive
    elif hasattr(msg, 'steering_angle') and hasattr(msg, 'speed'):
        flat_data['cmd_angle'] = float(msg.steering_angle)
        flat_data['cmd_speed'] = float(msg.speed)

    # Process standard primitive messages (Float32, Int32, String, etc.)
    elif hasattr(msg, 'data') and not isinstance(msg.data, (list, tuple, bytes)):
        # Remove the leading slash for column naming
        clean_topic = topic_name.strip('/')
        flat_data[clean_topic] = msg.data

    return flat_data



def is_rosbag(path):
    """Checks if a given path is a valid rosbag (mcap/db3 file or dir with metadata)."""
    if os.path.isfile(path) and path.endswith(('.mcap', '.db3')):
        return True
    if os.path.isdir(path) and os.path.exists(os.path.join(path, 'metadata.yaml')):
        return True
    return False

def find_rosbags(paths, recursive):
    """Parses input arguments to locate valid rosbags."""
    valid_bags = []
    for p in paths:
        if not os.path.exists(p):
            print(f"Warning: Path not found, skipping: {p}", file=sys.stderr)
            continue
            
        if recursive and os.path.isdir(p) and not is_rosbag(p):
            for root, dirs, files in os.walk(p):
                if 'metadata.yaml' in files:
                    valid_bags.append(root)
                    dirs.clear()  # Do not recurse inside a valid rosbag
                else:
                    for f in files:
                        if f.endswith('.mcap') or f.endswith('.db3'):
                            valid_bags.append(os.path.join(root, f))
        else:
            if is_rosbag(p):
                valid_bags.append(p)
            else:
                print(f"Warning: Not a valid rosbag (missing metadata.yaml or .mcap): {p}", file=sys.stderr)
                
    # Deduplicate while preserving order
    return list(dict.fromkeys(valid_bags))


def process_single_bag(bag_path, trigger_topics):
    print(f"\nProcessing bag: {bag_path}")
    
    if os.path.isdir(bag_path):
        meta_path = os.path.join(bag_path, 'metadata.yaml')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = yaml.safe_load(f)
                    if not meta or 'rosbag2_bagfile_information' not in meta or 'version' not in meta['rosbag2_bagfile_information']:
                        print(f"Error opening bag {bag_path}: metadata.yaml is malformed or missing key structure.", file=sys.stderr)
                        return [], set()
            except Exception as e:
                print(f"Error opening bag {bag_path}: Fast-fail on corrupted metadata.yaml: {e}", file=sys.stderr)
                return [], set()

    try:
        reader = get_reader(bag_path)
        type_map = extract_topic_types(reader)
    except Exception as e:
        print(f"Error opening bag {bag_path}: {e}", file=sys.stderr)
        return [], set()

    valid_triggers = [t for t in trigger_topics if t in type_map]
    if not valid_triggers:
        print(f"Error: None of the trigger topics {trigger_topics} were found in {bag_path} metadata.\nTopic information:", file=sys.stderr)
        try:
            for t_info in reader.get_metadata().topics_with_message_count:
                t_meta = t_info.topic_metadata
                print(f"                   Topic: {t_meta.name} | Type: {t_meta.type} | Count: {t_info.message_count}", file=sys.stderr)
        except Exception:
            pass
        return [], set()

    # State cache for Zero-Order Hold synchronization
    current_state = {}
    rows = []
    all_seen_columns = set(["timestamp"])

    print("Parsing messages...")
    message_count = 0

    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()
        message_count += 1

        if topic_name not in type_map:
            continue

        # Deserialize binary data into the Python message object
        msg_type = type_map[topic_name]
        msg = deserialize_message(data, msg_type)

        # Flatten the message into column key-value pairs
        flat_data = flatten_message(topic_name, msg)
        
        # Update our current state cache
        current_state.update(flat_data)
        all_seen_columns.update(flat_data.keys())

        # If this is the trigger topic, capture the state and append a new row
        if topic_name in trigger_topics:
            # Convert timestamp from nanoseconds to seconds for readability
            row_data = {"timestamp": timestamp / 1e9}
            row_data.update(current_state)
            rows.append(row_data)

    print(f"Parsed {message_count} messages. Generated {len(rows)} synchronized rows.")
    return rows, all_seen_columns


def write_csv(output_path, rows, all_seen_columns, keep_cols):
    if not rows:
        print(f"No rows to write for {output_path}. Skipping.")
        return
    if keep_cols:
        final_columns = ["timestamp"] + [col for col in keep_cols if col != "timestamp"]
    else:
        # Sort columns nicely (timestamp first, then commands, then others, then lidar)
        lidar_cols = sorted([c for c in all_seen_columns if c.startswith("lidar")], key=lambda x: int(x.replace("lidar[", "").replace("]", "")))
        cmd_cols = sorted([c for c in all_seen_columns if c.startswith("cmd_")])
        other_cols = sorted([c for c in all_seen_columns if c not in lidar_cols and c not in cmd_cols and c != "timestamp"])
        final_columns = ["timestamp"] + cmd_cols + other_cols + lidar_cols

    # Write to CSV
    print(f"Writing to {output_path}...")
    with open(output_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=final_columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def main():
    args = parse_args()
    bags = find_rosbags(args.bag_paths, args.recursive)
    
    if not bags:
        print("No valid rosbags found. Exiting.", file=sys.stderr)
        sys.exit(1)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for bag in bags:
            rows, seen_cols = process_single_bag(bag, args.trigger_topics)
            if rows:
                base_name = os.path.basename(os.path.normpath(bag))
                if base_name.endswith('.mcap') or base_name.endswith('.db3'):
                    base_name = os.path.splitext(base_name)[0]
                out_path = os.path.join(args.output_dir, f"{base_name}.csv")
                write_csv(out_path, rows, seen_cols, args.keep_cols)
    else:
        all_rows = []
        global_seen_cols = set()
        for bag in bags:
            rows, seen_cols = process_single_bag(bag, args.trigger_topics)
            all_rows.extend(rows)
            global_seen_cols.update(seen_cols)
            
        if all_rows:
            write_csv(args.output, all_rows, global_seen_cols, args.keep_cols)
    print("Done.")


if __name__ == "__main__":
    main()
