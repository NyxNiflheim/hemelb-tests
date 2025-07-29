# file: compare_outputs.py
# compare mask of inf and nan values, then compare numerical values

import sys
import os
import argparse
import numpy as np
import h5py

hemelb_python_tools_path = '/work/m24oc/m24oc/s2450341/hemelb/python-tools'
if hemelb_python_tools_path not in sys.path:
    sys.path.append(hemelb_python_tools_path)
try:
    from hlb.parsers.extraction import ExtractedProperty
except ImportError:
    print(f"ERROR: Cannot find the 'hlb' module. Please ensure the path is correct:")
    print(f"'{hemelb_python_tools_path}'")
    sys.exit(1)


def load_and_sort_xtr_data(xtr_path, timestep):
    # load and sort XTR data
    print(f"Loading and sorting XTR data from: {xtr_path} for timestep {timestep}")
    loader = ExtractedProperty(xtr_path)
    if timestep not in loader.times:
        raise ValueError(f"Timestep {timestep} not found in XTR file. Available: {loader.times}")
    snapshot = loader.GetByTimeStep(timestep)
    coord_field_name = 'grid' if 'grid' in snapshot.dtype.names else 'position'
    coords = snapshot[coord_field_name]
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    sorted_data = {}
    for field in snapshot.dtype.names:
        if field == 'id' or field == 'position':
            continue
        sorted_data[field] = snapshot[field][order]
    print(f"XTR data loaded and sorted.")
    return sorted_data

def load_and_sort_hdf5_data(h5_path, timestep):
    # load and sort HDF5 data
    print(f"Loading and sorting HDF5 data from: {h5_path} for timestep {timestep}")
    sorted_data = {}
    with h5py.File(h5_path, 'r') as f:
        group_name = f"step_{timestep}"
        if group_name not in f: raise ValueError(f"Group '{group_name}' not found in HDF5 file.")
        group = f[group_name]
        if 'geometry' not in group: raise ValueError("HDF5 group must contain a 'geometry' dataset.")
        
        coords = group['geometry'][:]
        order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
        
        sorted_data['grid'] = coords[order]

        for name in group.keys():
            if name == 'geometry':
                continue
            dataset = group[name]
            data = dataset[:]
            sorted_data[name] = data[order]
    print(f"HDF5 data loaded and sorted.")
    return sorted_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Numerically compare HemeLB's XTR and HDF5 outputs.")
    parser.add_argument("xtr_file", help="Path to the reference XTR (.dat) file.")
    parser.add_argument("h5_file", help="Path to the new HDF5 (.h5) file.")
    parser.add_argument("--timestep", type=int, required=True, help="The timestep to compare (e.g., 100).")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="Relative tolerance for float comparison.")
    args = parser.parse_args()

    xtr_data = load_and_sort_xtr_data(args.xtr_file, args.timestep)
    h5_data = load_and_sort_hdf5_data(args.h5_file, args.timestep)
    
    print("\n--- Starting Numerical Comparison ---")
    
    fields_to_compare = ['grid', 'pressure', 'shearstress', 'developed_velocity_field']
    total_mismatched_fields = 0

    for field_name in fields_to_compare:
        print(f"\nComparing field: '{field_name}'...")

        if field_name not in xtr_data or field_name not in h5_data:
            print(f"  ERROR: Field '{field_name}' not found in one of the files.")
            total_mismatched_fields += 1
            continue

        data_xtr = xtr_data[field_name]
        data_h5 = h5_data[field_name]
        
        is_mismatch = False
        
        if field_name == 'grid':
            data_xtr_int = data_xtr.astype(np.int64)
            data_h5_int = data_h5.astype(np.int64)
            if not np.array_equal(data_xtr_int, data_h5_int):
                is_mismatch = True
        else:
            data_xtr_flat = data_xtr.flatten()
            data_h5_flat = data_h5.flatten()
            
            # create masks for NaN and Inf values
            mask_xtr_nan = np.isnan(data_xtr_flat)
            mask_h5_nan = np.isnan(data_h5_flat)
            mask_xtr_inf = np.isinf(data_xtr_flat)
            mask_h5_inf = np.isinf(data_h5_flat)
            
            # compare NaN and Inf mask
            nan_pattern_ok = np.array_equal(mask_xtr_nan, mask_h5_nan)
            inf_pattern_ok = np.array_equal(mask_xtr_inf, mask_h5_inf)
            
            print(f"  NaN pattern matches: {nan_pattern_ok}")
            print(f"  Inf pattern matches: {inf_pattern_ok}")
            
            # compare numerical values of finite numbers
            # create a mask for valid (finite) numbers
            valid_mask = ~ (mask_xtr_nan | mask_h5_nan | mask_xtr_inf | mask_h5_inf)
            
            clean_xtr = data_xtr_flat[valid_mask]
            clean_h5 = data_h5_flat[valid_mask]
            
            numerical_ok = np.allclose(clean_xtr, clean_h5, rtol=args.tolerance, equal_nan=False)
            print(f"  Numerical values of finite numbers match: {numerical_ok}")
            
            if not (nan_pattern_ok and inf_pattern_ok and numerical_ok):
                is_mismatch = True

        if is_mismatch:
            print(f"  !!! MISMATCH FOUND in field '{field_name}'")
            total_mismatched_fields += 1
        else:
            print("  OK.")

    if total_mismatched_fields == 0:
        print("\nValidation Successful! All specified fields match numerically.")
    else:
        print(f"\nValidation Finished. Found mismatches in {total_mismatched_fields} fields.")

    print(f"  {np.sum(mask_xtr_nan)} NaNs in XTR, {np.sum(mask_h5_nan)} in HDF5")
    print(f"  {np.sum(mask_xtr_inf)} Infs in XTR, {np.sum(mask_h5_inf)} in HDF5")
    print(f"  Compared {np.sum(valid_mask)} finite elements")
    print(f"  Max abs diff = {np.max(np.abs(clean_xtr - clean_h5))}")
