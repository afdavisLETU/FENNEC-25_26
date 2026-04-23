import re
import os
import glob
import pandas as pd
import numpy as np
import math
import torch
import json
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

def get_pilot_labels(csv_dir, pilots):
    """
    Reads all filenames in a folder and returns 1D CG characterization labels

    Args:
        csv_dir (string): Directory of .csv files from which to get labels
    
    Returns:
        labels (list): A list of all the characterization labels

    """
    # the output labels list
    labels = []

    # Build a single regex pattern for all initials
    pattern = rf'clip\d+[A-Za-z]+_({"|".join(pilots)})_(AA|CG|FD)_(L|S)\.csv$'


    # --- CHECK FILEPATH ---
    if not os.path.isdir(csv_dir):
        raise FileNotFoundError(
            f"Error: Save path '{csv_dir}' not found. "
            f"Please create the directory before running the function."
        )

    # --- GET FILEPATHS ---
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))

    # --- MATCH AND EXTRACT INITIALS ---
    for file in csv_files:
        filename = os.path.basename(file)
        match = re.search(pattern, filename)
        if match:
            label = match.group(1)  # capture the initials (e.g., "JA", "BC")
            labels.append(label)
    print(f"got labels: {labels}")

    return labels


def get_gps(filepath):
    # --- FILE & FOLDER CHECKS ---
    if not os.path.isfile(filepath): # does .xlsx file exist?
        raise FileNotFoundError(
            f"Error: Input file '{filepath}' does not exist."
        )

    if not filepath.lower().endswith(".xlsx"): # is the file an .xlsx file?
        raise ValueError(
            f"Error: Input file '{filepath}' is not an .xlsx file."
        )

    inputfile = os.path.basename(filepath) #get the name of the xlsx file
    filename = inputfile[:-5] #remove the ".xlsx" from the end

    xl = pd.ExcelFile(filepath) #load the .xlsx into a pandas array (takes the longest)

    vars_of_interest = {"GPS": ["Lat", "Lng"]}

    extracted_data = {key: None for key in vars_of_interest} #stores only the designated data from each xl sheet

    #get the correct data from each sheet in the pandas array
    for sheet, variables in vars_of_interest.items():
        #make sure sheet exist in .xlsx
        if sheet not in xl.sheet_names:
            raise ValueError(
                f"Error: The sheet '{sheet}' was not found in {inputfile}. "
                f"Available sheets: {xl.sheet_names}"
            )
        
        df = xl.parse(sheet) #parse the correct sheet

        #make sure vars exist in sheet
        missing_cols = [col for col in variables if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Error: In sheet '{sheet}', the following columns are missing: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )        

        extracted_data[sheet] = df[variables].to_numpy(dtype=float) #save the designated data to extracted_data as a numpy array

        # extracted_data[sheet] = np.repeat(extracted_data[sheet], 2, axis=0).astype(float) #IMU freq. / RCOU/IN freq. = 400Hz / 10Hz = 40

    #stack all the data from each sheet into one single 2D array
    gps_data = np.hstack(list(extracted_data.values()))
    xl.close()
    return gps_data

def unpack_csv_dir(csv_dir):
    """
    Args:
        csv_dir: The path (including the folder name) of cleaned data
        
    Returns:
       list of numpy arrays, 1 per flight
    """
    
    if not os.path.isdir(csv_dir): # does savepath exist?
        raise FileNotFoundError(
            f"Error: Save path '{csv_dir}' not found. "
            f"Please create the directory before running the function."
        )
    # Paths
    clean_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    # SORTED() IS ESSENTIAL TO ENSURE FILES MATCH get_labels() LABELS
    all_data = []
    all_names = []

    #load all data so the scaler fits to the WHOLE data range
    for file in clean_files:
        all_names.append(os.path.basename(file)[4:11])
        df = pd.read_csv(file) #get csv data to PANDAS
        arr = df.to_numpy() #make pandas data numpy
        all_data.append(arr)

    return all_data, all_names