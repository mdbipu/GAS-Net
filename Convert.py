import numpy as np
import h5py
import os

# Path to your NPZ file on C drive
npz_path = r"E:/4th_draw/GAS-Net Paper/revision/GASNet/data/flickr30k_gas_label.npz"
h5_path  = r"E:/4th_draw/GAS-Net Paper/revision/GASNet/data/flickr30k_gas_label.h5"

# Load the NPZ file
data = np.load(npz_path)

# Create HDF5 file and write datasets
with h5py.File(h5_path, "w") as h5f:
    for key in data.files:
        h5f.create_dataset(key, data=data[key])
        print(f"Saved dataset: {key}")

print(f"Conversion complete! HDF5 file saved at: {h5_path}")
