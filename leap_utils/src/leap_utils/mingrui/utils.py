import os, sys
import pandas as pd
import numpy as np


# ----------------------------------------------
def check_and_create_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # If not, create the folder
        os.makedirs(folder_path)


# ----------------------------------------------
def saveDict(path, dict, csv=True, npy=True):
    directory = os.path.dirname(path)
    check_and_create_folder(directory)
    if csv:
        for key in list(dict.keys()):
            if not isinstance(dict[key], list):
                dict[key] = dict[key].tolist()
        df = pd.DataFrame(dict)
        df.to_csv(path + ".csv", index=False)
    if npy:
        np.save(path + ".npy", dict)


# ----------------------------------------------
def find_subdirectories(folder_path):
    subdirectories = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    return sorted(subdirectories)