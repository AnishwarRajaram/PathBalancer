import os
import numpy as np
from tqdm import tqdm # Install this for a nice progress bar: pip install tqdm
from nuscenes.nuscenes import NuScenes
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial


# Import your functions from the other files
# Ensure sampleToBEVgrid.py and groundTruthtoBEVgrid.py are in the same folder
from sampleToBEVgrid import get_bev_input
from groundTruthtoBEVgrid import get_bev_gt

# Find relative path
current_script_path = Path(__file__).resolve()
parent_folder = current_script_path.parent
data_path = parent_folder / "trainingData" / "v1.0-mini"

# Configuration
DATAROOT = data_path

VERSION = 'v1.0-mini'
OUTPUT_DIR_X = './processed_data/input_bev'
OUTPUT_DIR_Y = './processed_data/ground_truth'

SAMPLE_RANGE = (0,404)

def process_single_sample(sample_token, dataroot):
    """
    This function runs on a single CPU core.
    We re-initialize a local 'nusc' object inside because 
    nuScenes objects cannot be easily shared across processes.
    """
    # Local initialization for the worker process
    local_nusc = NuScenes(version=VERSION, dataroot=dataroot, verbose=False)
    
    try:
        x_data = get_bev_input(local_nusc, sample_token)
        y_data = get_bev_gt(local_nusc, sample_token)
        
        np.save(os.path.join(OUTPUT_DIR_X, f"{sample_token}_x.npy"), x_data)
        np.save(os.path.join(OUTPUT_DIR_Y, f"{sample_token}_y.npy"), y_data)
        return True
    except Exception as e:
        return f"Error {sample_token}: {e}"





def run_parallel_preprocessing():
    #make dirs if they don't exist
    os.makedirs(OUTPUT_DIR_X, exist_ok=True)
    os.makedirs(OUTPUT_DIR_Y, exist_ok=True)

    # 1. Get tokens and apply the slice
    main_nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
    
    start, end = SAMPLE_RANGE
    all_samples = main_nusc.sample[start:end]
    tokens = [s['token'] for s in all_samples]
    
    print(f"Processing samples from index {start} to {start + len(tokens)}...")

    # 2. Setup Workers
    # For Apple Silicon/Unix, 'fork' is efficient. 
    # For Windows, 'spawn' is used automatically.
    num_workers = max(1, cpu_count() - 1)
    worker_func = partial(process_single_sample, dataroot=DATAROOT)

    with Pool(num_workers) as p:
        # Using list() to force the generator to execute
        results = list(tqdm(p.imap(worker_func, tokens), total=len(tokens)))
        
        # Quick check for errors
        errors = [r for r in results if isinstance(r, str)]
        if errors:
            print(f"Encountered {len(errors)} errors. Check first error: {errors[0]}")

if __name__ == "__main__":
    run_parallel_preprocessing()
