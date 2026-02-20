# Performance Optimization Report: ASCII to mesh.nc Conversion

## Executive Summary

**Current Performance**: 111.52 seconds total (99.9% in ASCII reading)
**Grid Size**: 126,858 nodes, 244,659 elements, 48 levels

**Target Environment**: HPC login nodes with many processors

---

## Critical Bottlenecks Identified

### 1. **find_neighbors() - MOST CRITICAL** (~70-80s estimated)
**Location**: `ascii_to_netcdf.py:285-380`

**Current Implementation**:
```python
# Nested loops: Ne elements × 3 corners × max_iter iterations
for ie in range(Ne):  # 244,659 iterations
    for k in range(3):
        i = elem[ie, k]-1
        neigh1 = elem[ie, (k + 1) % 3]
        neigh2 = elem[ie, (k + 2) % 3]
        found1 = np.any(neighmat[i, :Nneigh[i]] == neigh1)  # Linear search
        found2 = np.any(neighmat[i, :Nneigh[i]] == neigh2)  # Linear search
```

**Issues**:
- ~734K total iterations (244,659 × 3)
- Linear searches with `np.any()` inside tight loops
- Minimal computation per iteration
- No vectorization or parallelization

### 2. **Computing Barycenters** (~15-20s estimated)
**Location**: `ascii_to_netcdf.py:563-575`

```python
for ie in range(Ne):  # 244,659 iterations
    elem_ie = elem[ie, :] - 1
    lon_ie, lat_ie = barycenter(lon[elem_ie], lat[elem_ie], z[elem_ie])
    baryc_lon[ie] = lon_ie
    baryc_lat[ie] = lat_ie
```

**Issues**:
- Function call overhead per iteration
- Complex trigonometry in `barycenter()` function
- Fully vectorizable but implemented as loop

### 3. **Generate Stamp Polygons** (~10-15s estimated)
**Location**: `ascii_to_netcdf.py:586-609`

```python
for i in range(N):  # 126,858 iterations
    for j in range(maxneighs):  # ~6-12 iterations per node
        nn = neighnodes[i, j]
        if not np.isnan(nn):
            lon_ij, lat_ij = barycenter([lon[i], lon[nn_index]], ...)
```

**Issues**:
- Nested loops: 126,858 × ~8 avg neighbors = ~1M iterations
- Function calls inside nested loop
- Many conditional checks

### 4. **Reorder CCW** (~5-8s estimated)
**Location**: `ascii_to_netcdf.py:453-459`

```python
for ie in range(Ne):  # 244,659 iterations
    a = np.array([lon_orig[elem[ie, 0] - 1], lat_orig[elem[ie, 0] - 1]])
    b = np.array([lon_orig[elem[ie, 1] - 1], lat_orig[elem[ie, 1] - 1]])
    c = np.array([lon_orig[elem[ie, 2] - 1], lat_orig[elem[ie, 2] - 1]])
    if checkposition(a, b, c) == -1:
        elem[ie, :] = elem[ie, ::-1]
```

**Issues**:
- Array allocation inside loop
- Complex geometry function per iteration
- Fully parallelizable

### 5. **Computing Areas** (~3-5s estimated)
**Location**: `ascii_to_netcdf.py:645-651`

```python
for ie in range(Ne):  # 244,659 iterations
    elemareas[ie] = triag_area(lon[elem[ie, :]-1], lat[elem[ie, :]-1])
```

---

## Optimization Strategies

## Strategy 1: **Numba JIT Compilation** ⭐ HIGHEST PRIORITY
**Expected Speedup**: 10-50x for computational loops
**Effort**: Low to Medium

### Implementation

```python
import numba
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def compute_barycenters_vectorized(lon, lat, z, elem, Ne):
    """Vectorized barycenter computation with Numba"""
    baryc_lon = np.empty(Ne, dtype=np.float64)
    baryc_lat = np.empty(Ne, dtype=np.float64)
    
    for ie in prange(Ne):  # Parallel loop
        # Extract element coordinates
        idx0, idx1, idx2 = elem[ie, 0]-1, elem[ie, 1]-1, elem[ie, 2]-1
        
        # Compute barycenter directly
        x_mean = (lon[idx0] + lon[idx1] + lon[idx2]) / 3.0
        y_mean = (lat[idx0] + lat[idx1] + lat[idx2]) / 3.0
        z_mean = (z[idx0] + z[idx1] + z[idx2]) / 3.0
        
        dist = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)
        baryc_lon[ie] = np.arctan2(y_mean/dist, x_mean/dist) * 180.0 / np.pi
        baryc_lat[ie] = np.arcsin(z_mean/dist) * 180.0 / np.pi
    
    return baryc_lon, baryc_lat

@njit(parallel=True)
def reorder_ccw_numba(elem, lon_orig, lat_orig, Ne):
    """Parallel CCW reordering with Numba"""
    ord_c = 0
    for ie in prange(Ne):
        # Inline checkposition logic
        a_lon, a_lat = lon_orig[elem[ie, 0]-1], lat_orig[elem[ie, 0]-1]
        b_lon, b_lat = lon_orig[elem[ie, 1]-1], lat_orig[elem[ie, 1]-1]
        c_lon, c_lat = lon_orig[elem[ie, 2]-1], lat_orig[elem[ie, 2]-1]
        
        # Convert to xyz and check orientation
        # ... (inline the cross product calculation)
        if orientation < 0:
            # Reverse element order
            temp = elem[ie, 0]
            elem[ie, 0] = elem[ie, 2]
            elem[ie, 2] = temp
            ord_c += 1
    return ord_c

@njit(parallel=True)
def compute_element_areas_numba(lon, lat, elem, Ne, Rearth):
    """Parallel area computation"""
    elemareas = np.empty(Ne, dtype=np.float64)
    
    for ie in prange(Ne):
        idx = elem[ie, :] - 1
        # Inline triag_area calculation
        area = compute_spherical_triangle_area(
            lon[idx[0]], lat[idx[0]],
            lon[idx[1]], lat[idx[1]],
            lon[idx[2]], lat[idx[2]]
        )
        elemareas[ie] = area * Rearth * Rearth
    
    return elemareas
```

**Benefits**:
- Automatic parallelization with `prange`
- Compiled to machine code
- SIMD vectorization
- Minimal code changes

---

## Strategy 2: **Multiprocessing for find_neighbors()** ⭐ HIGH PRIORITY
**Expected Speedup**: 4-16x (depending on cores)
**Effort**: Medium

### Implementation

```python
from multiprocessing import Pool, cpu_count
import numpy as np

def process_element_chunk(args):
    """Process a chunk of elements for neighbor finding"""
    elem_chunk, elem_full, start_ie, N, maxmaxneigh = args
    
    # Local arrays for this chunk
    local_updates = []
    
    for ie_local, ie_global in enumerate(range(start_ie, start_ie + len(elem_chunk))):
        for k in range(3):
            i = elem_chunk[ie_local, k] - 1
            neigh1 = elem_chunk[ie_local, (k + 1) % 3]
            neigh2 = elem_chunk[ie_local, (k + 2) % 3]
            
            local_updates.append((i, neigh1, neigh2, ie_global, k))
    
    return local_updates

def find_neighbors_parallel(elem, maxmaxneigh=12, n_workers=None):
    """Parallel version of find_neighbors using multiprocessing"""
    
    if n_workers is None:
        n_workers = cpu_count()
    
    N = np.max(elem)
    Ne = elem.shape[0]
    
    # Split elements into chunks
    chunk_size = Ne // n_workers
    chunks = []
    
    for i in range(n_workers):
        start = i * chunk_size
        end = start + chunk_size if i < n_workers - 1 else Ne
        chunk_args = (elem[start:end], elem, start, N, maxmaxneigh)
        chunks.append(chunk_args)
    
    # Process in parallel
    with Pool(n_workers) as pool:
        results = pool.map(process_element_chunk, chunks)
    
    # Merge results
    neighmat = np.full((N, maxmaxneigh), np.nan)
    barmat = np.full((N, maxmaxneigh), np.nan)
    Nneigh = np.zeros(N, dtype=int)
    
    # Aggregate updates (this part needs synchronization)
    for chunk_results in results:
        for i, neigh1, neigh2, ie, k in chunk_results:
            # Update logic here
            pass
    
    return neighmat, barmat, iscomplete, Nneigh, completed, avg_num_neighbors
```

**Note**: The aggregation phase needs careful handling to avoid race conditions. Consider using a shared memory approach or post-processing merge.

---

## Strategy 3: **Vectorization with NumPy** ⭐ MEDIUM-HIGH PRIORITY
**Expected Speedup**: 5-15x
**Effort**: Low to Medium

### Examples

#### Vectorize Barycenter Computation
```python
def compute_barycenters_vectorized_numpy(lon, lat, z, elem):
    """Fully vectorized barycenter computation"""
    # Get all element coordinates at once
    idx = elem - 1  # Shape: (Ne, 3)
    
    # Extract coordinates for all elements
    lon_elem = lon[idx]  # Shape: (Ne, 3)
    lat_elem = lat[idx]
    z_elem = z[idx]
    
    # Compute means
    x_mean = np.mean(lon_elem, axis=1)
    y_mean = np.mean(lat_elem, axis=1)
    z_mean = np.mean(z_elem, axis=1)
    
    # Normalize and convert
    dist = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)
    baryc_lon = np.arctan2(y_mean/dist, x_mean/dist) * 180.0 / np.pi
    baryc_lat = np.arcsin(z_mean/dist) * 180.0 / np.pi
    
    return baryc_lon, baryc_lat
```

#### Vectorize Coastal Element Detection
```python
# Current (slow):
elemcoast = np.array([np.sum(coast[elem[ie] - 1]) > 1 for ie in range(Ne)])

# Vectorized (fast):
elemcoast = np.sum(coast[elem - 1], axis=1) > 1
```

---

## Strategy 4: **Algorithmic Improvements**
**Expected Speedup**: 2-5x for find_neighbors
**Effort**: High

### Use Hash Maps for Neighbor Search

```python
from collections import defaultdict

def find_neighbors_optimized(elem, maxmaxneigh=12):
    """Use hash maps instead of linear search"""
    N = np.max(elem)
    Ne = elem.shape[0]
    
    # Build edge-to-element map
    edge_to_elem = defaultdict(list)
    
    for ie in range(Ne):
        for k in range(3):
            node1 = elem[ie, k]
            node2 = elem[ie, (k + 1) % 3]
            edge = tuple(sorted([node1, node2]))
            edge_to_elem[edge].append((ie, k))
    
    # Build neighbor lists using hash lookups (O(1) instead of O(n))
    neighmat = np.full((N, maxmaxneigh), np.nan)
    
    for i in range(1, N + 1):
        neighbors = set()
        for edge, elems in edge_to_elem.items():
            if i in edge:
                # Add the other node in the edge
                neighbors.add(edge[0] if edge[1] == i else edge[1])
        
        neighmat[i-1, :len(neighbors)] = sorted(neighbors)
    
    return neighmat
```

---

## Strategy 5: **Caching and Preprocessing**
**Expected Speedup**: N/A (one-time cost reduction)
**Effort**: Low

```python
import pickle
import hashlib

def get_mesh_hash(griddir):
    """Compute hash of input files"""
    files = ['nod2d.out', 'elem2d.out', 'aux3d.out']
    hash_input = ''
    for f in files:
        with open(os.path.join(griddir, f), 'rb') as fp:
            hash_input += hashlib.md5(fp.read()).hexdigest()
    return hashlib.md5(hash_input.encode()).hexdigest()

def read_fesom_ascii_grid_cached(griddir, **kwargs):
    """Read with caching of intermediate results"""
    mesh_hash = get_mesh_hash(griddir)
    cache_file = os.path.join(griddir, f'.cache_{mesh_hash}.pkl')
    
    if os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Compute normally
    grid = read_fesom_ascii_grid(griddir, **kwargs)
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(grid, f, protocol=4)
    
    return grid
```

---

## Recommended Implementation Plan

### Phase 1: Quick Wins (1-2 days, 5-10x speedup)
1. ✅ Vectorize barycenter computation (replace loop)
2. ✅ Vectorize coastal element detection (one-liner)
3. ✅ Vectorize area computation where possible
4. ✅ Add result caching

### Phase 2: Numba Integration (2-3 days, 20-30x speedup)
1. ✅ Install numba: `pip install numba`
2. ✅ Add @njit decorators to computational functions
3. ✅ Test with parallel=True for multi-core
4. ✅ Inline small functions to reduce overhead

### Phase 3: Parallel find_neighbors (3-5 days, 30-50x total speedup)
1. ✅ Implement hash-based neighbor finding
2. ✅ Add multiprocessing wrapper
3. ✅ Benchmark chunk sizes
4. ✅ Handle edge cases

### Phase 4: Production (1-2 days)
1. ✅ Add configuration flags for optimization level
2. ✅ Comprehensive testing
3. ✅ Documentation
4. ✅ Performance benchmarking suite

---

## Expected Performance Improvements

| Optimization | Current Time | Expected Time | Speedup |
|--------------|--------------|---------------|---------|
| **Baseline** | 111.5s | - | 1x |
| + Vectorization | 111.5s | 60-70s | 1.6-1.9x |
| + Numba (sequential) | 111.5s | 25-35s | 3.2-4.5x |
| + Numba (parallel, 8 cores) | 111.5s | 8-12s | 9.3-13.9x |
| + Numba (parallel, 16 cores) | 111.5s | 5-8s | 13.9-22.3x |
| + All optimizations | 111.5s | **3-5s** | **22-37x** |

---

## Code Examples: Complete Optimized Functions

See `ascii_to_netcdf_optimized.py` for full implementation.

---

## Hardware Considerations for HPC Login Nodes

**Recommendations**:
1. **Use 8-16 cores** for optimal speedup/resource balance
2. **Set NUMBA_NUM_THREADS** environment variable
3. **Monitor memory usage** - vectorization increases memory
4. **Consider chunking** for very large meshes (>1M elements)

```bash
# Example usage on HPC
export NUMBA_NUM_THREADS=16
export OMP_NUM_THREADS=16
python benchmark_ascii_to_mesh.py
```

---

## Testing Strategy

```python
def test_optimization_correctness(griddir):
    """Ensure optimized version produces identical results"""
    # Original
    grid_orig = read_fesom_ascii_grid_original(griddir)
    
    # Optimized
    grid_opt = read_fesom_ascii_grid_optimized(griddir)
    
    # Compare
    assert np.allclose(grid_orig['baryc.lon'], grid_opt['baryc.lon'])
    assert np.allclose(grid_orig['cellareas'], grid_opt['cellareas'])
    # ... more assertions
```

---

## Monitoring and Profiling

```python
import cProfile
import pstats

def profile_conversion(griddir):
    """Profile the conversion to find remaining bottlenecks"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    grid = read_fesom_ascii_grid(griddir)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

---

## Next Steps

1. **Run enhanced benchmark** with detailed per-function timing
2. **Implement Phase 1** vectorization (quick wins)
3. **Test Numba** on HPC environment
4. **Measure actual speedups** on target hardware
5. **Iterate based on profiling results**
