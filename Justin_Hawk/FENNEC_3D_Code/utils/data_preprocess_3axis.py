import os
import glob
import torch
import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

def extract_cg_code(filepath):
    filename = os.path.basename(filepath)
    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    if len(parts) < 2:
        raise ValueError(f"Invalid file format: {filename}")

    cg_code = parts[1]

    if len(cg_code) != 3:
        raise ValueError(f"Invalid CG code length: {filename}")

    return cg_code

def extract_axis_labels(filepath):
    code = extract_cg_code(filepath)   # example: FST
    x_char, y_char, z_char = code[0], code[1], code[2]

    x_map = {'F': 0, 'M': 1, 'A': 2}
    y_map = {'P': 0, 'C': 1, 'S': 2}
    z_map = {'T': 0, 'N': 1, 'B': 2}

    if x_char not in x_map:
        raise ValueError(f"Invalid X label '{x_char}' in {filepath}")
    if y_char not in y_map:
        raise ValueError(f"Invalid Y label '{y_char}' in {filepath}")
    if z_char not in z_map:
        raise ValueError(f"Invalid Z label '{z_char}' in {filepath}")

    return x_map[x_char], y_map[y_char], z_map[z_char]

class Flight:
    def __init__(self, filepath):
        self.filepath = filepath  #full path to CSV 
        self.filename = os.path.basename(filepath)
        self.cg_code = extract_cg_code(filepath)
        
        self.label_x, self.label_y, self.label_z = extract_axis_labels(filepath)
        self.data = self.load_data()

    def load_data(self):
        # load CSV,return as np.array - load vars of intrest?
        df = pd.read_csv(self.filepath)
        return df.values
    
class Flight_Objects:
    def __init__(self, csv_dir):
        self.csv_files = sorted(glob.glob(os.path.join(csv_dir,"*.csv")))
        self.flights = [Flight(f) for f in self.csv_files]

    def get_data_and_labels(self):
        #list of data and labels
        data = [f.data for f in self.flights]
        labels_x = [f.label_x for f in self.flights]
        labels_y = [f.label_y for f in self.flights]
        labels_z = [f.label_z for f in self.flights]
        return data, labels_x, labels_y, labels_z
    

# Flight splitter for train/test/val
def split_flights_by_class(flight_objects, train_per_class=3, val_per_class=0, seed=42):
    random.seed(seed)
    flights_by_class = {}

    for flight in flight_objects.flights:
        key = flight.cg_code
        flights_by_class.setdefault(key, []).append(flight)

    train_flights = []
    val_flights = []
    test_flights = []

    for key, flights in flights_by_class.items():
        if len(flights) < train_per_class + val_per_class + 1:
            raise ValueError(f"Not enough flights for label {key}")

        shuffled = flights.copy()
        random.shuffle(shuffled)

        train_flights.extend(shuffled[:train_per_class])
        val_flights.extend(shuffled[train_per_class:train_per_class + val_per_class])
        test_flights.extend(shuffled[train_per_class + val_per_class:train_per_class + val_per_class + 1])

    return train_flights, val_flights, test_flights

class FlightDataNormalizer:
    def __init__(self):
        self.scaler = None

    def fit(self, flights):
        # Fit the scaler to the train flight objects

        # Concatenate all flight data along axis 0 (timesteps)
        all_train_data = np.vstack([f.data for f in flights])
        self.scaler = StandardScaler()
        self.scaler.fit(all_train_data)
        print(f"Scaler fitted: mean shape {self.scaler.mean_.shape} var shape {self.scaler.var_.shape}")

        # Print mean and std for each feature
        print("Scaler fitted on training data:")
        for i, (mean, std) in enumerate(zip(self.scaler.mean_, np.sqrt(self.scaler.var_))):
            print(f"Feature {i}: mean = {mean:.4f}, std = {std:.4f}")

    def transform(self,flights):
        # Apply scaler to val/test data

        if self.scaler is None:
            raise RuntimeError("Scaler has not been fitted yet")
        
        for f in flights:
            f.data = self.scaler.transform(f.data)
        return flights
    
    def fit_transform(self, train_flights):
        self.fit(train_flights)
        return self.transform(train_flights)

class FlightDataSegmenter:
    def __init__(self, segment_length, max_segments_per_flight=None, seed=0):
        self.segment_length = segment_length
        self.max_segments_per_flight = max_segments_per_flight
        self.rng = np.random.default_rng(seed)

    def segment_flights(self, flights):
        segments, labels_x, labels_y, labels_z = [], [], [], []

        for f in flights:
            data = f.data
            n_timesteps = data.shape[0]

            cutoff = (n_timesteps // self.segment_length) * self.segment_length
            if cutoff == 0:
                continue

            data = data[:cutoff]
            segmented = np.split(data, cutoff // self.segment_length)

            if self.max_segments_per_flight is not None and len(segmented) > self.max_segments_per_flight:
                idx = self.rng.choice(len(segmented), size=self.max_segments_per_flight, replace=False)
                segmented = [segmented[i] for i in idx]

            segments.extend(segmented)
            labels_x.extend([f.label_x] * len(segmented))
            labels_y.extend([f.label_y] * len(segmented))
            labels_z.extend([f.label_z] * len(segmented))

        return (
            np.stack(segments),
            np.array(labels_x),
            np.array(labels_y),
            np.array(labels_z),
        )

# Data loaders
def make_dataloader(segments, labels_x, labels_y, labels_z, batch_size=32, shuffle=True):
    X_tensor = torch.tensor(segments, dtype=torch.float32)
    yx_tensor = torch.tensor(labels_x, dtype=torch.long)
    yy_tensor = torch.tensor(labels_y, dtype=torch.long)
    yz_tensor = torch.tensor(labels_z, dtype=torch.long)

    dataset = TensorDataset(X_tensor, yx_tensor, yy_tensor, yz_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader