#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark script for converting ASCII mesh files to mesh.nc
"""

import time
import sys
import os
from pyfesom2.ascii_to_netcdf import read_fesom_ascii_grid, write_mesh_to_netcdf

def benchmark_ascii_to_mesh(griddir, output_file=None):
    """
    Benchmark the conversion of ASCII mesh files to mesh.nc
    
    Parameters:
    -----------
    griddir : str
        Path to directory containing ASCII mesh files
    output_file : str, optional
        Output mesh.nc file path. If None, will use griddir/mesh_benchmark.nc
    """
    
    if output_file is None:
        output_file = os.path.join(griddir, 'mesh_benchmark.nc')
    
    print("=" * 80)
    print("BENCHMARK: ASCII to mesh.nc conversion")
    print("=" * 80)
    print(f"Input directory: {griddir}")
    print(f"Output file: {output_file}")
    print("=" * 80)
    print()
    
    # Step 1: Read ASCII grid files
    print("STEP 1: Reading ASCII grid files...")
    print("-" * 80)
    start_read = time.time()
    
    grid = read_fesom_ascii_grid(
        griddir=griddir,
        verbose=True
    )
    
    end_read = time.time()
    read_time = end_read - start_read
    
    print("-" * 80)
    print(f"✓ ASCII grid reading completed in {read_time:.2f} seconds")
    print()
    
    # Step 2: Write to netCDF
    print("STEP 2: Writing mesh to netCDF...")
    print("-" * 80)
    start_write = time.time()
    
    write_mesh_to_netcdf(
        grid,
        ofile=output_file,
        overwrite=True,
        verbose=True
    )
    
    end_write = time.time()
    write_time = end_write - start_write
    
    print("-" * 80)
    print(f"✓ NetCDF writing completed in {write_time:.2f} seconds")
    print()
    
    # Summary
    total_time = read_time + write_time
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"ASCII reading time:    {read_time:>10.2f} seconds ({read_time/total_time*100:>5.1f}%)")
    print(f"NetCDF writing time:   {write_time:>10.2f} seconds ({write_time/total_time*100:>5.1f}%)")
    print(f"Total conversion time: {total_time:>10.2f} seconds")
    print("=" * 80)
    print()
    
    # Grid statistics
    print("GRID STATISTICS")
    print("=" * 80)
    print(f"Number of 2D nodes:       {grid['N']:>10}")
    print(f"Number of elements:       {grid['Nelem']:>10}")
    if grid['Nlev'] is not None:
        print(f"Number of vertical levels: {grid['Nlev']:>10}")
        print(f"Total 3D nodes:           {grid['N3D']:>10}")
    print("=" * 80)
    print()
    
    print(f"✓ Output file created: {output_file}")
    print(f"  File size: {os.path.getsize(output_file) / (1024**2):.2f} MB")
    print()
    
    return {
        'read_time': read_time,
        'write_time': write_time,
        'total_time': total_time,
        'grid_info': {
            'n_nodes': grid['N'],
            'n_elements': grid['Nelem'],
            'n_levels': grid['Nlev'],
            'n_3d_nodes': grid['N3D']
        }
    }


if __name__ == '__main__':
    griddir = '/work/ab0246/a270092/input/fesom2/core2/'
    
    if not os.path.exists(griddir):
        print(f"Error: Directory {griddir} does not exist")
        sys.exit(1)
    
    benchmark_ascii_to_mesh(griddir)
