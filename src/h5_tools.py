import h5py
import hdf5plugin
import numpy as np


def get_c2_array(h5file, bin_number):
    """Access a c2 two-time correlation array by bin number."""
    key = f'xpcs/twotime/correlation_map/c2_{bin_number:05d}'
    return np.array(h5file[key])

def list_c2_bins(h5file):
    """Return list of available bin numbers."""
    group = h5file['xpcs/twotime/correlation_map']
    return sorted([int(k.split('_')[1]) for k in group.keys() if k.startswith('c2_')])

def print_h5_item(item, indent=''):
    """
    Recursively print the contents of an h5py group or dataset.
    
    Args:
    - item: The h5py group or dataset to print.
    - indent: A string of spaces used to indent nested items for better readability.
    """
    
    if isinstance(item, h5py.Group):  # Check if item is a group
        for key, subitem in item.items():
            print(f"{indent}/{key}")  # Print group name
            print_h5_item(subitem, indent + '    ')  # Recursively print contents of the group with additional indentation
    elif isinstance(item, h5py.Dataset):  # Check if item is a dataset
        print(f"{indent}[Dataset] Shape: {item.shape}, Type: {item.dtype}")
        # To print actual data, uncomment the line below. Be cautious with large datasets.
        # print(item[:])

