import os
import glob
import re

def get_3D_CG_labels(csv_dir):
    if not os.path.isdir(csv_dir):
        raise FileNotFoundError(f"Directory not found: {csv_dir}")

    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    labels = []

    patterns = {
        "AAA": r'^\d+[GB]_AAA_(L|H|S)_\d+\.csv',
        "AAB": r'^\d+[GB]_AAB_(L|H|S)_\d+\.csv',
        "AAC": r'^\d+[GB]_AAC_(L|H|S)_\d+\.csv',

        "ABA": r'^\d+[GB]_ABA_(L|H|S)_\d+\.csv',
        "ABB": r'^\d+[GB]_ABB_(L|H|S)_\d+\.csv',
        "ABC": r'^\d+[GB]_ABC_(L|H|S)_\d+\.csv',

        "ACA": r'^\d+[GB]_ACA_(L|H|S)_\d+\.csv',
        "ACB": r'^\d+[GB]_ACB_(L|H|S)_\d+\.csv',
        "ACC": r'^\d+[GB]_ACC_(L|H|S)_\d+\.csv',

        "BAA": r'^\d+[GB]_BAA_(L|H|S)_\d+\.csv',
        "BAB": r'^\d+[GB]_BAB_(L|H|S)_\d+\.csv',
        "BAC": r'^\d+[GB]_BAC_(L|H|S)_\d+\.csv',   
        "BBA": r'^\d+[GB]_BBA_(L|H|S)_\d+\.csv',
        "BBB": r'^\d+[GB]_BBB_(L|H|S)_\d+\.csv',
        "BBC": r'^\d+[GB]_BBC_(L|H|S)_\d+\.csv',
        "BCA": r'^\d+[GB]_BCA_(L|H|S)_\d+\.csv',
        "BCB": r'^\d+[GB]_BCB_(L|H|S)_\d+\.csv',
        "BCC": r'^\d+[GB]_BCC_(L|H|S)_\d+\.csv',

        "CAA": r'^\d+[GB]_CAA_(L|H|S)_\d+\.csv',
        "CAB": r'^\d+[GB]_CAB_(L|H|S)_\d+\.csv',
        "CAC": r'^\d+[GB]_CAC_(L|H|S)_\d+\.csv',
        "CBA": r'^\d+[GB]_CBA_(L|H|S)_\d+\.csv',
        "CBB": r'^\d+[GB]_CBB_(L|H|S)_\d+\.csv',
        "CBC": r'^\d+[GB]_CBC_(L|H|S)_\d+\.csv',
        "CCA": r'^\d+[GB]_CCA_(L|H|S)_\d+\.csv',
        "CCB": r'^\d+[GB]_CCB_(L|H|S)_\d+\.csv',
        "CCC": r'^\d+[GB]_CCC_(L|H|S)_\d+\.csv',
    }

    for file in csv_files:
        filename = os.path.basename(file)
        matched = False

        for label, pattern in patterns.items():
            if re.search(pattern, filename):
                labels.append(label)
                matched = True
                break  # <-- ENSURES ONLY ONE LABEL PER FILE

        if not matched:
            raise ValueError(f"ERROR: File does not match any pattern → {filename}")

    return labels

