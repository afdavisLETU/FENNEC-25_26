# Wills Kookogey with help from Micah Yarbrough and Claude
# 4/3/26

import os
import pandas as pd
import numpy as np
import re
import json
import glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Micah Yarbrough & Wills Kookogey
# 4/3/26
# This function will extract and clean up data from .xlsx files, saving them as .csv's
def data_cleaner_FID(filepath, savepath, overwrite= False, skip= False, varspath= "vars_of_interest.json", downsample= True):
    """
    Preprocesses .xlsx files into fennec question-usefull .csv files.

    Args:
        filepath (string): The .xlsx file to process.
        savepath (string): The folder to save the .csv file.
        overwrite (bool): Skips the overwrite checker if true.
        skip (bool): Skips duplicate files instead of checking or overwriting if true.
        varspath (string): The vars-of-interest.json path. Defaults to same folder as THIS script.
        downsample (bool): If true, it will downsample to the lowest sample rate, if false it will upsample to the highest

    Relies on the vars_of_interest.json file to determine what data is wanted
    """

    # --- FILE & FOLDER CHECKS ---
    if not os.path.isfile(filepath): # does .xlsx file exist?
        raise FileNotFoundError(
            f"Error: Input file '{filepath}' does not exist."
        )

    if not filepath.lower().endswith(".xlsx"): # is the file an .xlsx file?
        raise ValueError(
            f"Error: Input file '{filepath}' is not an .xlsx file."
        )

    if not os.path.isdir(savepath): # does savepath exist?
        raise FileNotFoundError(
            f"Error: Save path '{savepath}' not found. "
            f"Please create the directory before running the function."
        )

    if not os.path.isfile(varspath): # does vars_of_interest.json exist?
        raise FileNotFoundError(
            f"Error: Vars-of-interest file '{varspath}' not found. "
            f"Ensure the JSON file is in the same folder as this script OR pass the filepath via arg: varspath =\" \"."
        )


    inputfile = os.path.basename(filepath) #get the name of the xlsx file
    filename = inputfile[:-5] #remove the ".xlsx" from the end
    
    # --- OVERWRITE CHECKER ---
    if (overwrite == False): #skip if overwrite was set to True
        #check savepath to see if the .xlsx file has already been processed
        for csvfile in os.listdir(savepath):
            if os.path.basename(csvfile) == f"{filename}.csv":
                if(skip == True):
                    print(f"{inputfile} skipped due to existing duplicate.")
                    return None
                
                #if a match is found, prompt the user before overwriting the file
                user_input = ""
                while (user_input != "y") and (user_input != "n"):
                    user_input = input("ARE YOU SURE YOU WANT TO OVERWRITE THIS FILE? (y,n)-->")
                if user_input == "n":
                    print(f"{inputfile} not processed due to user input.")
                    return None
    
    # --- PREPROCESSING ---
    """
    For each sheet, we want to take the relevant data at each timestamp
       and package it together in an 2D array[x][y] where x is each timestamp and y is each datatype

        [[GyrX0, GyrY0, ..., AccZ0],
         [GyrX1, GyrY1, ..., AccZ1],
         [GyrX2, GyrY2, ..., AccZ2], ...]

        Then the arrays for each sheet get combined so EVERY datatype is stored at each timestamp.
        That combined array gets saved as a .csv file.
    """

    xl = pd.ExcelFile(filepath) #load the .xlsx into a pandas array (takes the longest)

    # read the vars_of_interest file
    with open(varspath, "r") as f:
        vars_of_interest = json.load(f) #convert json file to dict

    extracted_data = {key: None for key in vars_of_interest} #stores only the designated data from each xl sheet

    # get the correct data from each sheet in the pandas array
    for sheet, variables in vars_of_interest.items():
        #make sure sheet exists in .xlsx
        if sheet not in xl.sheet_names:
            raise ValueError(
                f"Error: The sheet '{sheet}' was not found in {inputfile}. "
                f"Available sheets: {xl.sheet_names}"
            )
        
        df = xl.parse(sheet) # parse the correct sheet

        # make sure vars exist in sheet
        missing_cols = [col for col in variables if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Error: In sheet '{sheet}', the following columns are missing: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )        

        extracted_data[sheet] = df[variables].to_numpy(dtype=float) # save the designated data to extracted_data as a numpy array
        
    # FREQUENCY CORRECTION
    lengths = [len(arr) for arr in extracted_data.values()] # get total timesteps for each sheet
    min_len = min(lengths) # find shortest sheet
    ratios = [round(l / min_len) for l in lengths]

    # Scale each sheet according to its scaling ratio
    for i, (sheet_name, sheet) in enumerate(extracted_data.items()):
        ratio = ratios[i]
        sheet = sheet[::ratio]
        # Downsample
        if downsample == True and ratio != 1:
            extracted_data[sheet_name] = sheet  # reassign if you want to keep the change
            print(f"Sheet {sheet_name}, downsampled by {ratio}")
        # Upsample
        elif downsample == False:
            ratio = max(ratios)/ratio
            if ratio != 1:
                extracted_data[sheet_name] = np.repeat(extracted_data[sheet_name], ratio, axis=0).astype(float)
                print(f"Sheet {sheet_name}, upsampled by {ratio}")

    # THIS IS SPECIFIC TO FID FOR LEAK LABELING
    # if RCIN C6 or C7 (leak switches) are in vars of interest at end, find the index of leak start in extracted_data,
    # then remove C6 and C7 from data
    if "RCIN" in vars_of_interest and "C6" in vars_of_interest["RCIN"] and "C7" in vars_of_interest["RCIN"]:
        c6_index = vars_of_interest["RCIN"].index("C6")
        c7_index = vars_of_interest["RCIN"].index("C7")
        # if C6 goes below 1500 label leak start (right tank started leaking)
        if extracted_data["RCIN"][:, c6_index].min() < 1500:
            leak_start = int(np.where(extracted_data["RCIN"][:, c6_index] < 1500)[0][0]) # Find index of leak start

        # else if C7 goes above 1500 label leak start (left tank started leaking)
        elif extracted_data["RCIN"][:, c7_index].max() > 1500:
            leak_start = int(np.where(extracted_data["RCIN"][:, c7_index] > 1500)[0][0]) # Find index of leak start
        
        else:
            leak_start = None
        
        # Delete in reverse index order so the first deletion doesn't shift the second index
    for idx in sorted([c6_index, c7_index], reverse=True):
        extracted_data["RCIN"] = np.delete(extracted_data["RCIN"], idx, axis=1)

    # Recalculate the lengths
    lengths = [len(arr) for arr in extracted_data.values()]
    min_len = min(lengths)
 
    # --- LENGTH CORRECTION ---
    # Truncate all arrays to the minimum length
    for sheet in extracted_data: 
        extracted_data[sheet] = extracted_data[sheet][:min_len] 
    lengths = [len(arr) for arr in extracted_data.values()]

    # stack all the data from each sheet into one single 2D array
    csv_data = np.hstack(list(extracted_data.values()))

    # --- SAVE AS .CSV ---
    df = pd.DataFrame(csv_data)
    new_path = os.path.join(savepath, inputfile.replace('xlsx', 'csv')) # Create new path
    df.to_csv(new_path, index=False, encoding='utf_8') # Save to new path

    print(f"{inputfile} processed and saved to {savepath} as {filename}.csv")
    xl.close()
    return leak_start

# Micah Yarbrough and Wills Kookogey
# 10/21/25
# This function will calls the data cleaner for every .xlsx file in a given directory
def folder_cleaner_FID(excel_dir, savepath, test_flight_names=None, overwrite = False, skip = False, varspath = "vars_of_interest.json", downsample= True):
    """
    Preprocesses a folder of .xlsx files into fennec question-usefull .csv files.

    Args:
        excel_dir (string): The folder of .xlsx files to process.
        savepath (string): The folder to save the .csv file.
        test_flight_names (list of strings): The list of test flight names to be saved in a separate folder.
        overwrite (bool): Skips the overwrite checker if true.
        skip (bool): Skips duplicate files instead of checking or overwriting if true.
        varspath (string): The vars-of-interest.json path. Defaults to same folder as THIS script.

    Relies on the vars_of_interest.json file to determine what data is wanted
    """
    
    # load existing train_val leak starts if they exist
    train_val_leak_starts_path = os.path.join(savepath, "leak_starts.json")
    if os.path.isfile(train_val_leak_starts_path):
        with open(train_val_leak_starts_path, "r") as f:
            train_val_leak_starts_dict = json.load(f)
    else:
        train_val_leak_starts_dict = {}  # keyed by filename, value is int or None
        
    # load existing test leak starts if they exist
    test_leak_starts_path = os.path.join(savepath, "test_flights/leak_starts.json")
    if os.path.isfile(test_leak_starts_path):
        with open(test_leak_starts_path, "r") as f:
            test_leak_starts_dict = json.load(f)
    else:
        test_leak_starts_dict = {}  # keyed by filename, value is int or None
    
    # --- FILE & FOLDER CHECKS ---
    if not os.path.isdir(excel_dir): # does savepath exist?
        raise FileNotFoundError(
            f"Error: Save path '{excel_dir}' not found. "
            f"Please create the directory before running the function."
        )
    
    # --- CLEAN FOLDER ---
    # for each file in the folder, run data_cleaner
    for file in os.listdir(excel_dir):
        filepath = os.path.join(excel_dir, file)
        if filepath.lower().endswith(".xlsx"):
            
            # if filename matches name in test list, save to test folder instead of train/val folder
            filename = os.path.splitext(os.path.basename(filepath))[0]
            if test_flight_names is not None and filename in test_flight_names:
                
                # create test folder if it doesn't exist
                test_savepath = os.path.join(savepath, "test_flights")
                os.makedirs(test_savepath, exist_ok=True)
                
                if not os.path.isdir(test_savepath):
                    raise FileNotFoundError(
                        f"Error: Save path '{test_savepath}' not found. "
                        f"Please create the test_flights directory before running the function."
                    )
                    
                result = data_cleaner_FID(filepath, test_savepath, overwrite, skip, varspath, downsample)

                if result is None:  # file was skipped
                    leak_start = test_leak_starts_dict.get(filename, None)  # recall from saved dict
                    if leak_start is None:
                        print(f"Warning: No saved leak start for skipped file {filename}")
                else:
                    leak_start = result
                    test_leak_starts_dict[filename] = leak_start  # save/update the entry

                # after the loop, save the updated dict
                with open(test_leak_starts_path, "w") as f:
                    json.dump(test_leak_starts_dict, f, indent=2)
            
            else:
                result = data_cleaner_FID(filepath, savepath, overwrite, skip, varspath, downsample)
                if result is None:  # file was skipped
                    leak_start = train_val_leak_starts_dict.get(filename, None)  # recall from saved dict
                    if leak_start is None:
                        print(f"Warning: No saved leak start for skipped file {filename}")
                else:
                    leak_start = result
                    train_val_leak_starts_dict[filename] = leak_start  # save/update the entry

                # after the loop, save the updated dict
                with open(train_val_leak_starts_path, "w") as f:
                    json.dump(train_val_leak_starts_dict, f, indent=2)
    
    return True



# Wills Kookogey
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
    leakstarts = []

    # Regex patterns for reading 2024-2025 1D and 2D CG flight data files
    patternL = r'^\d+[A-Za-z]_(L)\.csv$'
    patternR = r'^\d+[A-Za-z]_(R)\.csv$'
    patternLR = r'^\d+[A-Za-z]_(LR)\.csv$'
    patternNONE = r'^\d+[A-Za-z]_(NONE)\.csv$'

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
        if re.search(patternLR, filename):   # check LR first, before L
            labels.append("LR")
        elif re.search(patternL, filename):
            labels.append("L")
        elif re.search(patternR, filename):
            labels.append("R")
        elif re.search(patternNONE, filename):
            labels.append("NONE")
        
        # take extension off filename
        filename = os.path.splitext(filename)[0]
        
        # load existing train_val leak starts from json
        leak_starts_path = os.path.join(csv_dir, "leak_starts.json")
        if os.path.isfile(leak_starts_path):
            with open(leak_starts_path, "r") as f:
                leak_starts_dict = json.load(f)
        else:
            leak_starts_dict = {}  # keyed by filename, value is int or None
        
        # find filename in leak_starts json and append to leakstart list
        if filename in leak_starts_dict:
            leakstarts.append(int(leak_starts_dict[filename]))
        
        # if filename is NONE (so is not in leakstart list) append a 0 to leakstarts
        if filename not in leak_starts_dict:
            leakstarts.append(0)

    return labels, leakstarts




# Wills Kookogey & Micah Yarbrough
# 02-17-26
# This function segments the data into the number of timesteps desired and labels it
# It returns a dictionary
def segment_label_split_FID(train_data_input, test_data_input, train_labels_in, test_labels_in, train_val_leakstarts=0, test_leakstarts=0, sequence_length=10, train_split=0.8, shuffle=True):
    """
    Labels data into dataset dictionary
    output shape is (sequence, timestep, feature)

    Args:
        train_data (list): List of numpy arrays, 1 per file, all for training
        validate_data (list): List of numpy arrays, 1 per file, all for validating
        test_data (list): List of numpy arrays, 1 per file, all for testing
        input_labels (list): List of characterization lables, 1 per file (should correspond to input_data)
    
    Returns:
        output (dict): 3 labels: "Training_Set", "Validation_Set", and "Testing_Set"
            each set has the follwing labels: "sets" and "labels" 
                - "sets" : list of sets, corresponds to "labels"
                - "labels" : list of labels, corresponds to "sets"
            ex: output["Training_Set"]["sets"] 

    """
    
    # temp arrays
    train_segments = []
    train_labels = []
    validate_segments = []
    validate_labels = []
    test_segments = []
    test_labels = []
    
    # --- ERROR CHECKS ---
    if len(train_data_input) != len(train_labels_in):
        raise ValueError(
            f"Length mismatch: got {len(train_data_input)} train arrays but {len(train_labels_in)} labels. "
            "They must be the same length."
        )
    
    if len(test_data_input) != len(test_labels_in):
        raise ValueError(
            f"Length mismatch: got {len(test_data_input)} test arrays but {len(test_labels_in)} labels. "
            "They must be the same length."
        )

    # --- SPLIT ARRAYS ---
    
    # a function to segment and label data, used to avoid code repetition
    def segment_helper(data_input, labels_input, leak_starts, timesteps, shuffle):
        segments = []
        labels = []
        flight_lengths = []
        
        # for each numpy array (flight) in list, segment into segments of length = timesteps
        for index, array in enumerate(data_input):
            # find the maximum number of timesteps that can be cut
            cutoff = (len(array) // timesteps) * timesteps
            array = array[:cutoff]
            
            #split the array up into segments
            num_segments = len(array) // timesteps
            segmented_arrays = array.reshape(num_segments, timesteps, array.shape[1])
            segments.append(segmented_arrays)
            
            # label segments before leakstart as NONE
            if leak_starts[index] > 0:
                num_leak_segments = leak_starts[index] // timesteps
                labels.extend(["NONE"]*num_leak_segments)
            
            else:
                num_leak_segments = 0
            
            # label segments after leakstart with the correct label
            labels.extend([labels_input[index]]*(len(segmented_arrays) - num_leak_segments))
            
            # append the length of the flight in segments to flight_lengths to help when plotting results later
            flight_lengths.append(num_segments)
        
        segments = np.concatenate(segments, axis=0)
        labels = np.array(labels)
        
        # check if user wants all segments in order or shuffled
        if shuffle:
            index = np.random.permutation(len(labels))
            print("Segments shuffled")
        else:
            index = np.arange(len(labels))
            print("Segments not shuffled")
        
        return segments[index], labels[index], flight_lengths
    
    # segment and label each category of data
    train_val_segments, train_val_labels, _ = segment_helper(train_data_input, train_labels_in, train_val_leakstarts, sequence_length, shuffle=shuffle)
    test_segments, test_labels, test_flight_lengths = segment_helper(test_data_input, test_labels_in, test_leakstarts, sequence_length, shuffle=False)

    # split the training segments into train and validate
    train_end = int(len(train_val_labels)*train_split)
    validate_segments = train_val_segments[train_end:]
    validate_labels = train_val_labels[train_end:]
    train_segments = train_val_segments[:train_end]
    train_labels = train_val_labels[:train_end]

    # define output dict
    output = {
        "Training_Set": {"sets": train_segments, "labels": train_labels},
        "Validation_Set": {"sets": validate_segments, "labels": validate_labels},
        "Testing_Set": {"sets": test_segments, "labels": test_labels}
    }

    # Print out completion message and the number of sets in each category
    print("All data segmented and labeled!")
    print(f"Training_Sets: {len(output['Training_Set']['labels'])}")
    print(f"Validation_Sets: {len(output['Validation_Set']['labels'])}")
    print(f"Testing_Sets: {len(output['Testing_Set']['labels'])}")

    # return split data
    return output, test_flight_lengths

# Glory to the Father, and to the Son, and to the Holy Spirit: as
# it was in the beginning, is now, and will be for ever. Amen.

def segment_label_split_sliding_FID(train_data_input, test_data_input, train_labels_in, test_labels_in, train_val_leakstarts=0, test_leakstarts=0, sequence_length=10, train_split=0.8, shuffle=True, stride=None):
    """
    Labels data into dataset dictionary
    output shape is (sequence, timestep, feature)

    Args:
        train_data (list): List of numpy arrays, 1 per file, all for training
        validate_data (list): List of numpy arrays, 1 per file, all for validating
        test_data (list): List of numpy arrays, 1 per file, all for testing
        input_labels (list): List of characterization lables, 1 per file (should correspond to input_data)
    
    Returns:
        output (dict): 3 labels: "Training_Set", "Validation_Set", and "Testing_Set"
            each set has the follwing labels: "sets" and "labels" 
                - "sets" : list of sets, corresponds to "labels"
                - "labels" : list of labels, corresponds to "sets"
            ex: output["Training_Set"]["sets"] 

    """
    
    # temp arrays
    train_segments = []
    train_labels = []
    validate_segments = []
    validate_labels = []
    test_segments = []
    test_labels = []
    
    # --- ERROR CHECKS ---
    if len(train_data_input) != len(train_labels_in):
        raise ValueError(
            f"Length mismatch: got {len(train_data_input)} train arrays but {len(train_labels_in)} labels. "
            "They must be the same length."
        )
    
    if len(test_data_input) != len(test_labels_in):
        raise ValueError(
            f"Length mismatch: got {len(test_data_input)} test arrays but {len(test_labels_in)} labels. "
            "They must be the same length."
        )

    # --- SPLIT ARRAYS ---
    
    # a function to segment and label data, used to avoid code repetition
    def segment_helper(data_input, labels_input, leak_starts, timesteps, shuffle, stride=None):
        segments = []
        labels = []
        flight_lengths = []
        
        if stride is None:
            stride = timesteps  # default: non-overlapping (original behavior)
        
        for index, array in enumerate(data_input):
            flight_segments = []
            flight_labels = []
            
            for start in range(0, len(array) - timesteps + 1, stride):
                end = start + timesteps
                segment = array[start:end]
                
                # determine label based on where this window falls relative to leakstart
                leakstart = leak_starts[index]
                
                if leakstart > 0:
                    # window is entirely before leak: NONE
                    if end <= leakstart:
                        seg_label = "NONE"
                    
                    # window is entirely after leak: true label
                    elif start >= leakstart:
                        seg_label = labels_input[index]
                    
                    # window straddles the leak boundary: mixed
                    else:
                        # DROP: skip ambiguous boundary windows during training
                        # (for test data you may want to keep them — see note below)
                        continue
                
                else:
                    # NONE flights: label everything as the file label
                    seg_label = labels_input[index]
                
                flight_segments.append(segment)
                flight_labels.append(seg_label)
            
            segments.append(np.array(flight_segments))
            labels.extend(flight_labels)
            flight_lengths.append(len(flight_segments))
        
        segments = np.concatenate(segments, axis=0)
        labels = np.array(labels)
        
        if shuffle:
            idx = np.random.permutation(len(labels))
            print("Segments shuffled")
        else:
            idx = np.arange(len(labels))
            print("Segments not shuffled")
        
        return segments[idx], labels[idx], flight_lengths
    
    # segment and label each category of data
    train_val_segments, train_val_labels, _ = segment_helper(train_data_input, train_labels_in, train_val_leakstarts, sequence_length, shuffle=shuffle, stride=stride)
    test_segments, test_labels, test_flight_lengths = segment_helper(test_data_input, test_labels_in, test_leakstarts, sequence_length, shuffle=False, stride=None) # for test data, keep all segments (including boundary ones) to evaluate model performance on them

    # split the training segments into train and validate
    train_end = int(len(train_val_labels)*train_split)
    validate_segments = train_val_segments[train_end:]
    validate_labels = train_val_labels[train_end:]
    train_segments = train_val_segments[:train_end]
    train_labels = train_val_labels[:train_end]

    # define output dict
    output = {
        "Training_Set": {"sets": train_segments, "labels": train_labels},
        "Validation_Set": {"sets": validate_segments, "labels": validate_labels},
        "Testing_Set": {"sets": test_segments, "labels": test_labels}
    }

    # Print out completion message and the number of sets in each category
    print("All data segmented and labeled!")
    print(f"Training_Sets: {len(output['Training_Set']['labels'])}")
    print(f"Validation_Sets: {len(output['Validation_Set']['labels'])}")
    print(f"Testing_Sets: {len(output['Testing_Set']['labels'])}")

    # return split data
    return output, test_flight_lengths

# Glory to the Father, and to the Son, and to the Holy Spirit: as
# it was in the beginning, is now, and will be for ever. Amen. 