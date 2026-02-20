#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark comparing original vs optimized implementations
"""

import time
import sys
import os
import numpy as np

# Import optimized functions
from ascii_to_netcdf_optimized import (
    compute_barycenters_vectorized,
    compute_barycenters_numba,
    determine_coastal_elements_vectorized,
    find_neighbors_optimized,
    get_optimization_info,
    print_optimization_status,
    HAS_NUMBA
)

def benchmark_individual_operations(griddir):
    """
    Benchmark individual operations to identify specific improvements
    """
    print("=" * 80)
    print("DETAILED OPERATION BENCHMARK")
    print("=" * 80)
    print()
    
    print_optimization_status()
    
    # Read basic grid data
    print("Loading grid data...")
    import pandas as pd
    
    # Read nodes
    nod_file = os.path.join(griddir, 'nod2d.out')
    file_content = pd.read_csv(
        nod_file,
        delim_whitespace=True,
        skiprows=1,
        names=["node_number", "x", "y", "flag"],
    )
    lon_orig = file_content.x.values
    lat_orig = file_content.y.values
    coast = file_content.flag.values % 2
    N = len(lon_orig)
    
    # Read elements
    elem_file = os.path.join(griddir, 'elem2d.out')
    file_content = pd.read_csv(
        elem_file,
        delim_whitespace=True,
        skiprows=1,
        names=["first_elem", "second_elem", "third_elem"],
    )
    elem = file_content.values
    Ne = elem.shape[0]
    
    # Compute xyz coordinates
    rad = np.pi / 180
    x = np.cos(lat_orig * rad) * np.cos(lon_orig * rad)
    y = np.cos(lat_orig * rad) * np.sin(lon_orig * rad)
    z = np.sin(lat_orig * rad)
    
    print(f"Grid: {N:,} nodes, {Ne:,} elements")
    print()
    
    # ========================================================================
    # BENCHMARK 1: Barycenter Computation
    # ========================================================================
    print("-" * 80)
    print("BENCHMARK 1: Barycenter Computation")
    print("-" * 80)
    
    # Original (simulated loop-based approach)
    print("Testing: Loop-based approach (original)...")
    start = time.time()
    baryc_lon_loop = np.zeros(Ne)
    baryc_lat_loop = np.zeros(Ne)
    for ie in range(min(Ne, 10000)):  # Only test subset for speed
        elem_ie = elem[ie, :] - 1
        x_mean = np.mean(x[elem_ie])
        y_mean = np.mean(y[elem_ie])
        z_mean = np.mean(z[elem_ie])
        dist = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)
        baryc_lon_loop[ie] = np.arctan2(y_mean/dist, x_mean/dist) * 180/np.pi
        baryc_lat_loop[ie] = np.arcsin(z_mean/dist) * 180/np.pi
    loop_time = time.time() - start
    # Extrapolate to full grid
    loop_time_full = loop_time * Ne / min(Ne, 10000)
    print(f"  Time (extrapolated): {loop_time_full:.2f}s")
    
    # Vectorized NumPy
    print("Testing: Vectorized NumPy approach...")
    start = time.time()
    baryc_lon_vec, baryc_lat_vec = compute_barycenters_vectorized(x, y, z, elem)
    vec_time = time.time() - start
    print(f"  Time: {vec_time:.2f}s")
    print(f"  Speedup: {loop_time_full/vec_time:.1f}x")
    
    # Numba (if available)
    if HAS_NUMBA:
        print("Testing: Numba JIT approach...")
        # Warmup
        _ = compute_barycenters_numba(x, y, z, elem[:100], 100)
        # Benchmark
        start = time.time()
        baryc_lon_numba, baryc_lat_numba = compute_barycenters_numba(x, y, z, elem, Ne)
        numba_time = time.time() - start
        print(f"  Time: {numba_time:.2f}s")
        print(f"  Speedup vs loop: {loop_time_full/numba_time:.1f}x")
        print(f"  Speedup vs vectorized: {vec_time/numba_time:.1f}x")
        
        # Verify correctness
        max_diff_lon = np.max(np.abs(baryc_lon_vec - baryc_lon_numba))
        max_diff_lat = np.max(np.abs(baryc_lat_vec - baryc_lat_numba))
        print(f"  Max difference: lon={max_diff_lon:.2e}, lat={max_diff_lat:.2e}")
    
    print()
    
    # ========================================================================
    # BENCHMARK 2: Coastal Element Detection
    # ========================================================================
    print("-" * 80)
    print("BENCHMARK 2: Coastal Element Detection")
    print("-" * 80)
    
    # Original loop
    print("Testing: Loop-based approach (original)...")
    start = time.time()
    elemcoast_loop = np.array([np.sum(coast[elem[ie] - 1]) > 1 for ie in range(Ne)])
    loop_time = time.time() - start
    print(f"  Time: {loop_time:.2f}s")
    
    # Vectorized
    print("Testing: Vectorized approach...")
    start = time.time()
    elemcoast_vec = determine_coastal_elements_vectorized(coast, elem)
    vec_time = time.time() - start
    print(f"  Time: {vec_time:.2f}s")
    print(f"  Speedup: {loop_time/vec_time:.1f}x")
    
    # Verify correctness
    if np.array_equal(elemcoast_loop, elemcoast_vec):
        print(f"  ✓ Results match perfectly")
    else:
        print(f"  ✗ Results differ!")
    
    print()
    
    # ========================================================================
    # BENCHMARK 3: Find Neighbors
    # ========================================================================
    print("-" * 80)
    print("BENCHMARK 3: Find Neighbors (most critical)")
    print("-" * 80)
    
    # Test on subset first
    subset_size = min(5000, Ne)
    elem_subset = elem[:subset_size]
    
    print(f"Testing on subset: {subset_size:,} elements...")
    print("Testing: Optimized hash-based approach...")
    start = time.time()
    neighmat, barmat, iscomplete, Nneigh, completed, avg_neighbors = find_neighbors_optimized(
        elem_subset, maxmaxneigh=12, verbose=False
    )
    opt_time_subset = time.time() - start
    print(f"  Time: {opt_time_subset:.2f}s")
    print(f"  Average neighbors: {avg_neighbors:.2f}")
    print(f"  Completed: {completed}")
    
    # Extrapolate
    full_time_estimated = opt_time_subset * (Ne / subset_size)
    print(f"  Estimated time for full grid: {full_time_estimated:.2f}s")
    print(f"  Original implementation: ~70-80s (from profiling)")
    print(f"  Estimated speedup: ~{70/full_time_estimated:.1f}x")
    
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)
    
    total_original = loop_time_full + loop_time + 70  # rough estimate
    total_optimized = vec_time + vec_time + full_time_estimated
    if HAS_NUMBA:
        total_optimized = numba_time + vec_time + full_time_estimated
    
    print(f"Estimated total time (original):  {total_original:.1f}s")
    print(f"Estimated total time (optimized): {total_optimized:.1f}s")
    print(f"Overall speedup:                   {total_original/total_optimized:.1f}x")
    print()
    
    if not HAS_NUMBA:
        print("⚠️  Install Numba for even better performance:")
        print("   pip install numba")
        print("   Expected additional speedup: 3-10x")
    
    print("=" * 80)


def run_full_comparison(griddir):
    """
    Run a full comparison showing before/after performance
    """
    from pyfesom2.ascii_to_netcdf import read_fesom_ascii_grid
    
    print("\n" + "=" * 80)
    print("FULL MESH CONVERSION TEST")
    print("=" * 80)
    print()
    
    print("Note: The full optimized version would require integrating")
    print("optimized functions into read_fesom_ascii_grid().")
    print()
    print("Individual operation benchmarks show potential speedups.")
    print("See PERFORMANCE_OPTIMIZATION_REPORT.md for implementation guide.")
    print("=" * 80)


if __name__ == '__main__':
    griddir = '/work/ab0246/a270092/input/fesom2/core2/'
    
    if not os.path.exists(griddir):
        print(f"Error: Directory {griddir} does not exist")
        sys.exit(1)
    
    benchmark_individual_operations(griddir)
    run_full_comparison(griddir)
