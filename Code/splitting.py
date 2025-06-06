import os
import h5py
import numpy as np

# Paths and constants
BASE_DIR = "/scratch/tsingh65/Simba100-1_143_AbsSpectra_2500"
RAW_FILES_TEMPLATE = os.path.join(BASE_DIR, "spectra_Simba100_z0.1_n300d2-sample_localized_COS-G130M_OVI_{i}-of-14.hdf5")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

NUM_CORES = 100  # Number of cores/tasks

def split_hdf5_file(file_index):
    """
    Splits the given HDF5 file into NUM_CORES chunks and saves them into the corresponding core directories,
    preserving all datasets as specified.
    """
    # Paths for this group
    hdf5_file = RAW_FILES_TEMPLATE.format(i=file_index)
    group_dir = os.path.join(RESULTS_DIR, f"group_{file_index}")
    os.makedirs(group_dir, exist_ok=True)
    
    # Open the HDF5 file
    with h5py.File(hdf5_file, "r") as f:
        total_spectra = f["flux"].shape[0]  # Total number of spectra
        chunk_size = total_spectra // NUM_CORES

        # Process each chunk
        for core_id in range(NUM_CORES):
            core_dir = os.path.join(group_dir, f"core_{core_id}")
            os.makedirs(core_dir, exist_ok=True)

            start = core_id * chunk_size
            end = (core_id + 1) * chunk_size if core_id != NUM_CORES - 1 else total_spectra

            # Save the chunk into a new HDF5 file
            core_file = os.path.join(core_dir, f"core_{core_id}.hdf5")
            with h5py.File(core_file, "w") as core_f:
                for key in f.keys():
                    data = f[key]
                    if key in ["EW_OVI_1031", "EW_OVI_1037"]:  # 1D datasets split by spectra
                        core_f.create_dataset(key, data=data[start:end])
                    elif key == "flux":  # 2D dataset split by spectra
                        core_f.create_dataset(key, data=data[start:end, :])
                    elif key == "ray_pos":  # 2D dataset split by spectra
                        core_f.create_dataset(key, data=data[start:end, :])
                    elif key in ["tau_OVI_1031", "tau_OVI_1037"]:  # 2D dataset split by spectra
                        core_f.create_dataset(key, data=data[start:end, :])
                    elif key == "ray_dir":  # 1D dataset kept as is
                        core_f.create_dataset(key, data=data)
                    elif key == "ray_total_dl":  # Scalar dataset kept as is
                        core_f.create_dataset(key, data=data)
                    elif key == "wave":  # 1D dataset kept as is
                        core_f.create_dataset(key, data=data)
                    else:
                        raise ValueError(f"Unexpected key {key} found in the HDF5 file.")

            print(f"Saved chunk {core_id} for group {file_index} to {core_file}")

def main():
    for i in range(0, 14):  # Process files 1 to 13
        print(f"Processing group {i}")
        split_hdf5_file(i)
        print(f"Finished processing group {i}")

if __name__ == "__main__":
    main()