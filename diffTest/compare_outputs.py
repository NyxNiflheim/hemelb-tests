# file: compare_outputs.py

import sys
import os
import argparse
import numpy as np
import h5py
# Path to HemeLB Python tools
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
    """load xtr file, and return sorted data dictionary."""
    print(f"Loading and sorting XTR data from: {xtr_path} for timestep {timestep}")
    loader = ExtractedProperty(xtr_path)
    if timestep not in loader.times:
        raise ValueError(f"Timestep {timestep} not found in XTR file. Available: {loader.times}")
    snapshot = loader.GetByTimeStep(timestep)
    
    coord_field_name = 'grid' if 'grid' in snapshot.dtype.names else 'position'
    if coord_field_name not in snapshot.dtype.names:
        raise ValueError(f"Could not find coordinate field ('grid' or 'position') in XTR data.")
    
    coords = snapshot[coord_field_name]
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    
    sorted_data = {}
    for field in snapshot.dtype.names:
        if field == 'id' or field == 'position': # ignore 'id' and 'position' fields
            continue
        sorted_data[field] = snapshot[field][order]
    
    print(f"XTR data loaded and sorted.")
    return sorted_data

def load_and_sort_hdf5_data(h5_path, timestep):
    """load HDF5 file, and return sorted data dictionary."""
    print(f"Loading and sorting HDF5 data from: {h5_path} for timestep {timestep}")
    sorted_data = {}
    with h5py.File(h5_path, 'r') as f:
        group_name = f"step_{timestep}"
        if group_name not in f: raise ValueError(f"Group '{group_name}' not found in HDF5 file.")
        group = f[group_name]
        if 'geometry' not in group: raise ValueError("HDF5 group must contain a 'geometry' dataset.")
        
        coords = group['geometry'][:]
        order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
        
        # rename 'geometry' to 'grid' for consistency
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
    results_summary = []
    total_mismatched_fields = 0

    for field_name in fields_to_compare:
        print(f"\nComparing field: '{field_name}'...")

        if field_name not in xtr_data or field_name not in h5_data:
            print(f"  ERROR: Field '{field_name}' not found in one of the files.")
            total_mismatched_fields += 1
            continue

        data_xtr = xtr_data[field_name]
        data_h5 = h5_data[field_name]
        
        num_elements = data_xtr.size
        num_mismatches = 0
        
        if field_name == 'grid':
            # grid field is always integer coordinates
            data_xtr_int = data_xtr.astype(np.int64)
            data_h5_int = data_h5.astype(np.int64)
            comparison_array = np.equal(data_xtr_int, data_h5_int)
        else:
            # use tolerance for float fields
            # flatten the arrays to compare all elements
            data_xtr_flat = data_xtr.flatten()
            data_h5_flat = data_h5.flatten()
            
            # XDR files may contain Inf values, handle them appropriately
            if field_name == 'shearstress':
                data_xtr_flat = np.nan_to_num(data_xtr_flat, nan=-1.0, posinf=-1.0, neginf=-1.0)
            
            comparison_array = np.isclose(data_xtr_flat, data_h5_flat, rtol=args.tolerance, equal_nan=True)

        num_mismatches = np.sum(~comparison_array)
        
        if num_mismatches > 0:
            total_mismatched_fields += 1
            print(f"  !!! MISMATCH FOUND: {num_mismatches} elements differ.")
            
            # get all indices of mismatches
            mismatch_indices = np.where(~comparison_array)
            unique_mismatched_rows = np.unique(mismatch_indices[0])
            
            print(f"  --- Showing values for first 19 mismatched sites: ---")
            for i, row_idx in enumerate(unique_mismatched_rows[:19]):
                print(f"    Site Index {row_idx}: XDR = {data_xtr[row_idx]}, HDF5 = {data_h5[row_idx]}")
        else:
            print("  OK.")
        
        results_summary.append({
            "field": field_name,
            "total_compared": num_elements,
            "mismatches": num_mismatches
        })
    
    # print summary of results
    print("\n\n--- Comparison Summary ---")
    print(f"{'Field Name':<30} | {'Mismatched Elements':<20} | {'Total Elements Compared':<25}")
    print("-" * 80)
    for res in results_summary:
        print(f"{res['field']:<30} | {res['mismatches']:<20} | {res['total_compared']:<25}")
    
    print("-" * 80)
    if total_mismatched_fields == 0:
        print("\nValidation Successful! All specified fields match numerically.")
    else:
        print(f"\nValidation Finished. Found mismatches in {total_mismatched_fields} fields.")