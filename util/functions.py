from util import *
import pickle as pkl
from pathlib import Path


def process_isear_dataset():
    """
    Loads ISEAR dataset, processes it, and saves it to a Pickled file in 
    dictionary format. 
    If Pickled file already exists, loads and returns dataset.
    INPUTS:
        None
    RETURNS:
        dataset (Dict): dataset with format {sentence : emotion (7 classes)}.
    """
    # If processed ISEAR dataset file doesn't exist, do processing
    if not Path(isear_processed_path).is_file():
        dataset = {}
        with open(isear_raw_path, "r") as f:
            for line in f.readlines():
                
                # If line contains error symbols, skip
                if error_symbol in line:
                    continue
                
                # Preprocess line
                for stop_symbol in stop_symbols:
                    line = line.replace(stop_symbol, "")
                if "/" in line:
                    line = line.replace("/", " / ")
                if "'" in line:
                    if "o'clock" in line:
                        pass
                    else:
                        line = line.replace("'", " '")
                line = line.lower().split("---")
                
                # Append to dataset
                #! Duplicate sentences exist with differing emotions. Dictionary is 
                #! used to set unique sentence to one emotion.
                dataset[line[2]] = line[1]
        
        # Dump dataset (dictionary) to Pickle file
        pkl.dump(dataset, isear_processed_path)
    else:
        dataset = pkl.load(isear_processed_path)
    
    return dataset