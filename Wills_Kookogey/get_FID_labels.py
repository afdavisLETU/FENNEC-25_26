import os
import pandas as pd
import numpy as np
import re
import json
import glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# WIlls Kookogey
# 11/4/25
# Reads all filenames in a folder and returns FID characterization labels
def get_FID_labels(csv_dir):
    """
    Reads all filenames in a folder and returns FID characterization labels

    Args:
        csv_dir (string): Directory of .csv files from which to get labels
    
    Returns:
        labels (list): A list of all the characterization labels

    """
    # the output labels list
    labels = []

    # Regex patterns for reading 2024-2025 1D and 2D CG flight data files
    patternL = r'^\d+G_(L).csv$'
    patternR = r'^\d+G_(R).csv$'
    patternLR = r'^\d+G_(LR).csv$'
    patternNONE = r'^\d+G_(NONE).csv$'

    # --- CHECK FILEPATH ---
    if not os.path.isdir(csv_dir): # does savepath exist?
        raise FileNotFoundError(
            f"Error: Save path '{csv_dir}' not found. "
            f"Please create the directory before running the function."
        )

    # --- GET FILEPATHS ---
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    # SORTED() IS ESSENTIAL TO ENSURE FILES MATCH normalize() data

    # append label of each file to labels list
    for file in csv_files:
        filename = os.path.basename(file)
        if re.search(patternL, filename):
            labels.append("L")
        if re.search(patternR, filename):
            labels.append("R")
        if re.search(patternLR, filename):
            labels.append("LR")
        if re.search(patternNONE, filename):
            labels.append("NONE")
    
    return labels
