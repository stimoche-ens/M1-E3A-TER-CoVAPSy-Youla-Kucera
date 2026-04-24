import os
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import csv
import numpy as np
import glob

files = glob.glob("rosbag2*.csv") 

for file in files:
    try:
        df = pd.read_csv(file)
    except Exception as e:
        print(f"Skipping {file} due to read error: {e}")
        continue

    if df.empty:
        print(f"Skipping {file}: No data rows found.")
        continue

    # 2. Extract Lidar data (columns 3 to end)
    # iloc[:, 3:] selects all rows and columns from index 3 onwards
    lidars_0 = df.iloc[:, 3:].to_numpy(dtype=float)
    
    # Check if lidar_data contains only NaNs to avoid RuntimeWarnings
    if np.isnan(lidars_0).all():
        print(f"Skipping {file} due to no valid lidar readings.")
        continue


    lidars_0 = np.roll(lidars_0, shift=180, axis=1)

    df.iloc[:, 3:] = lidars_0

    out_dirname=os.path.dirname(file)
    out_basename="shift_"+os.path.basename(file)
    out_fullname=os.path.join(out_dirname,out_basename)
    df.to_csv(out_fullname, index=False)
    print(f"Processed: {out_fullname}")