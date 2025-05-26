# extract_feature.py

import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os


def load_data(folder_path):
  all_files = glob.glob(os.path.join(folder_path, '*.csv'))
  processed_data = []

  for filename in all_files:
    print("Processing file:", filename)
    df = pd.read_csv(filename)
    X_processed, y_processed = preprocess_data(df)
    if X_processed.size > 0: 
      processed_data.append((X_processed, y_processed))

  X_combined = np.concatenate([data[0] for data in processed_data],
                              axis=0) if processed_data else np.array([])
  y_combined = np.concatenate([data[1] for data in processed_data],
                              axis=0) if processed_data else np.array([])

  return X_combined, y_combined


def preprocess_data(frame):
  frame['day'] = pd.to_datetime(frame['time']).dt.day
  frame['month'] = pd.to_datetime(frame['time']).dt.month
  frame['year'] = pd.to_datetime(frame['time']).dt.year
  frame["time"] = pd.to_datetime(frame["time"])
  frame["time_in_hour"] = frame["time"].dt.hour
  frame["time_in_minute"] = frame["time"].dt.minute
  frame["time_in_seconds"] = frame["time"].dt.second
  frame['plate'] = frame['plate'].astype('int64')

  X = frame[[
      'longitude', 'latitude', 'status', 'day', 'month', 'time_in_hour',
      'time_in_minute', 'time_in_seconds'
  ]].values
  Y = frame['plate']

  scaler = StandardScaler()

  X_reshaped = []
  y_reshaped = []

  for plate in frame['plate'].unique():
    plate_frame = frame[frame['plate'] == plate]
    X_plate = plate_frame[[
        'longitude', 'latitude', 'status', 'day', 'month', 'time_in_hour',
        'time_in_minute', 'time_in_seconds'
    ]].values

    if len(X_plate) > 0:
      X_scaled = scaler.fit_transform(X_plate)
      num_chunks = len(X_scaled) // 100

      for i in range(num_chunks):
        chunk = X_scaled[i * 100:(i + 1) * 100]
        X_reshaped.append(chunk)
        y_reshaped.append(plate)

      if len(X_scaled) % 100 != 0:
        last_chunk = X_scaled[num_chunks * 100:]
        last_chunk_padded = np.pad(last_chunk,
                                   ((0, 100 - len(last_chunk)), (0, 0)),
                                   mode='constant')
        X_reshaped.append(last_chunk_padded)
        y_reshaped.append(plate)

  X_reshaped = np.array(X_reshaped)
  y_reshaped = np.array(y_reshaped)

  print("X_reshaped shape:", X_reshaped.shape)
  print("y_reshaped shape:", y_reshaped.shape)

  return X_reshaped, y_reshaped
