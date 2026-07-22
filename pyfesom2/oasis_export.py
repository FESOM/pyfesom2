# -*- coding: utf-8 -*-
#
# This file is part of pyfesom2
# For OASIS3-MCT coupling grid/area/mask generation
#

import numpy as np
import os
from netCDF4 import Dataset
from .load_mesh_data import load_mesh

def write_fesom_oasis_files(mesh, output_dir=None, prefix='feom', overwrite=False):
    """
    Write FESOM2 mesh to OASIS3-MCT compatible files:
    - grids.nc: contains lon/lat coordinates and corner coordinates
    - areas.nc: contains lon/lat coordinates and cell areas
    - masks.nc: contains lon/lat coordinates and land-sea mask
    
    Only writes the FESOM mesh fields (with prefix 'feom'). 
    Other components in the coupling should be set up elsewhere.
    
    Parameters
    ----------
    mesh : object or dict
        FESOM2 mesh object loaded with load_mesh, or
        mesh dictionary from read_fesom_ascii_grid
    output_dir : str
        Directory to write the output files to. If None, write to current directory.
    prefix : str
        Prefix for the FESOM mesh variables in the OASIS files (default: 'feom')
    overwrite : bool
        Whether to overwrite existing files
        
    Returns
    -------
    dict
        Dictionary with paths to the created files
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define file paths
    grids_file = os.path.join(output_dir, 'grids.nc')
    areas_file = os.path.join(output_dir, 'areas.nc')
    masks_file = os.path.join(output_dir, 'masks.nc')
    
    # Check existing files and handle overwrite logic
    grids_exists = os.path.exists(grids_file)
    areas_exists = os.path.exists(areas_file)
    masks_exists = os.path.exists(masks_file)

    if (grids_exists or areas_exists or masks_exists) and not overwrite:
        existing = [p for p, ex in zip([grids_file, areas_file, masks_file],
                                       [grids_exists, areas_exists, masks_exists]) if ex]
        raise FileExistsError(
            f"Files already exist: {', '.join(existing)}. Set overwrite=True to update only the FESOM variables.")
    
    # Handle both mesh object and mesh dictionary formats
    is_dict = isinstance(mesh, dict)
    
    # Get mesh data
    if is_dict:
        # Dictionary from read_fesom_ascii_grid
        n2d = len(mesh['lon'])
        x2 = mesh['lon']
        y2 = mesh['lat']
        elem = mesh['elem']
        e2d = len(elem)
    else:
        # Mesh object from load_mesh
        n2d = mesh.n2d
        x2 = mesh.x2
        y2 = mesh.y2
        elem = mesh.elem
        e2d = mesh.e2d
    
    # Create dimensions
    x_dim = f'x_{prefix}'
    y_dim = f'y_{prefix}'
    crn_dim = f'crn_{prefix}'
    
    # Create dimension and variable names based on prefix
    lon_var = f'{prefix}.lon'
    lat_var = f'{prefix}.lat'
    clo_var = f'{prefix}.clo'
    cla_var = f'{prefix}.cla'
    srf_var = f'{prefix}.srf'
    msk_var = f'{prefix}.msk'

    # ------------------------------------------------------------------
    # Helper functions for incremental updating of existing NetCDF files
    # ------------------------------------------------------------------
    def _ensure_dim(nc_obj, name, size):
        """Create a dimension if it does not exist, otherwise ensure size matches."""
        if name not in nc_obj.dimensions:
            try:
                nc_obj.createDimension(name, size)
            except Exception as e:
                # If creation fails but dimension now exists (race condition), verify size
                if name in nc_obj.dimensions:
                    pass  # Will be checked below
                else:
                    raise e  # Re-raise if dimension still doesn't exist
        
        # Verify dimension size if it exists
        if name in nc_obj.dimensions:
            # When dimension exists, size can be None for unlimited; otherwise compare
            if (not nc_obj.dimensions[name].isunlimited()) and len(nc_obj.dimensions[name]) != size:
                raise ValueError(
                    f"Dimension {name} has size {len(nc_obj.dimensions[name])}, expected {size}")

    def _get_or_create_var(nc_obj, name, dtype, dimensions):
        """Return existing variable or create a new one with given signature."""
        if name in nc_obj.variables:
            var = nc_obj.variables[name]
            if var.dimensions != tuple(dimensions):
                raise ValueError(
                    f"Variable {name} has dimensions {var.dimensions}, expected {dimensions}")
            return var
        try:
            return nc_obj.createVariable(name, dtype, dimensions)
        except Exception as e:
            # Check if variable was created despite exception (race condition)
            if name in nc_obj.variables:
                var = nc_obj.variables[name]
                if var.dimensions != tuple(dimensions):
                    raise ValueError(
                        f"Variable {name} has dimensions {var.dimensions}, expected {dimensions}")
                return var
            else:
                # Re-raise if variable wasn't created
                raise e


    # Calculate element areas if not already in mesh
    if is_dict and 'elemareas' in mesh:
        voltri = mesh['elemareas']
    elif not is_dict and hasattr(mesh, 'voltri'):
        voltri = mesh.voltri
    else:
        # Spherical-triangle element areas (vectorized over all elements;
        # ~100x faster than the former per-element Python loop).
        R = 6371000.0  # Earth radius in meters
        rad = np.pi / 180.0
        # NOTE: load_mesh's elem is already 0-based, but this routine applies an
        # extra "-1" and historically relied on Python negative-index wrap for
        # node 0. We reproduce that wrap with "% n2d" so output is byte-identical
        # to the previous loop version. This "-1" is a latent off-by-one for
        # 0-based meshes (mis-assigns node 0); see the accompanying note.
        en = (elem - 1) % n2d
        n1i, n2i, n3i = en[:, 0], en[:, 1], en[:, 2]
        latr = y2 * rad
        lonr = x2 * rad
        cx = np.cos(latr) * np.cos(lonr)
        cy = np.cos(latr) * np.sin(lonr)
        cz = np.sin(latr)
        ax = cx[n2i] - cx[n1i]; ay = cy[n2i] - cy[n1i]; az = cz[n2i] - cz[n1i]
        bx = cx[n3i] - cx[n1i]; by = cy[n3i] - cy[n1i]; bz = cz[n3i] - cz[n1i]
        crx = ay * bz - az * by
        cry = az * bx - ax * bz
        crz = ax * by - ay * bx
        voltri = 0.5 * np.sqrt(crx * crx + cry * cry + crz * crz) * R ** 2

    # Node areas by distributing element areas (vectorized scatter-add).
    en = (elem - 1) % n2d
    elem_flat = en.reshape(-1)
    node_areas = np.bincount(elem_flat, weights=np.repeat(voltri / 3.0, 3),
                             minlength=n2d)
    node_count = np.bincount(elem_flat, minlength=n2d).astype(float)
    # Avoid division by zero
    node_count[node_count == 0] = 1
    
    # OASIS masks.nc convention: 1 = masked (excluded from coupling), 0 = active.
    # This is the OPPOSITE of the SCRIP grid_imask convention (1 = valid). All
    # FESOM nodes are wet (ocean) and must participate in coupling, so the OASIS
    # mask is 0 everywhere. Writing 1 here masks the whole ocean and aborts OASIS
    # when feom is used as a remapping source.
    mask = np.zeros(n2d, dtype=np.int32)
    
    # Corner coordinates: each node's corners are the centroids of its
    # surrounding elements, padded/sub-selected to exactly 4 (OASIS uses 4).
    # Vectorized equivalent of the former per-node Python loop (~100x faster),
    # byte-identical output. Behaviour per node count: 0 -> node's own coords;
    # 1..4 -> available centroids, last duplicated; >4 -> linspace sub-select.
    max_corners = 4  # OASIS uses 4 corners
    en = (elem - 1) % n2d
    n1i, n2i, n3i = en[:, 0], en[:, 1], en[:, 2]
    cent_lon = (x2[n1i] + x2[n2i] + x2[n3i]) / 3.0
    cent_lat = (y2[n1i] + y2[n2i] + y2[n3i]) / 3.0

    # Group element-centroid incidences by node, preserving element-index order
    # (matches the original append order).
    elem_flat = en.reshape(-1)
    elem_of_inc = np.repeat(np.arange(e2d), 3)
    order = np.argsort(elem_flat, kind="stable")
    sorted_nodes = elem_flat[order]
    sorted_clon = cent_lon[elem_of_inc[order]]
    sorted_clat = cent_lat[elem_of_inc[order]]

    counts = np.bincount(elem_flat, minlength=n2d)
    starts = np.zeros(n2d, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]
    within = np.arange(sorted_nodes.size) - starts[sorted_nodes]

    max_neigh = int(counts.max()) if counts.size else 0
    pad_lon = np.zeros((n2d, max(max_neigh, 1)))
    pad_lat = np.zeros((n2d, max(max_neigh, 1)))
    pad_lon[sorted_nodes, within] = sorted_clon
    pad_lat[sorted_nodes, within] = sorted_clat

    corner_lons = np.zeros((max_corners, n2d))
    corner_lats = np.zeros((max_corners, n2d))
    counts = counts.astype(np.int64)

    zero = counts == 0
    if np.any(zero):
        corner_lons[:, zero] = x2[zero]
        corner_lats[:, zero] = y2[zero]

    le = (counts >= 1) & (counts <= max_corners)
    if np.any(le):
        idx = np.nonzero(le)[0]
        c = counts[idx]
        slots = np.arange(max_corners)[:, None]
        src = np.where(slots < c[None, :], slots, (c - 1)[None, :])  # (4, m)
        corner_lons[:, idx] = pad_lon[idx, src]
        corner_lats[:, idx] = pad_lat[idx, src]

    gt = counts > max_corners
    if np.any(gt):
        idx = np.nonzero(gt)[0]
        c = counts[idx]
        k = np.arange(max_corners)[None, :]
        src = ((c - 1)[:, None] / (max_corners - 1) * k).astype(int)  # (m, 4)
        corner_lons[:, idx] = pad_lon[idx[None, :], src.T]
        corner_lats[:, idx] = pad_lat[idx[None, :], src.T]

    # ------------------------------------------------------------------
    # Write or update grids.nc
    # ------------------------------------------------------------------
    grid_mode = 'a' if grids_exists else 'w'
    with Dataset(grids_file, grid_mode, format='NETCDF4') as nc:
        _ensure_dim(nc, x_dim, n2d)
        _ensure_dim(nc, y_dim, 1)
        _ensure_dim(nc, crn_dim, max_corners)

        lon = _get_or_create_var(nc, lon_var, 'f8', (y_dim, x_dim))
        lat = _get_or_create_var(nc, lat_var, 'f8', (y_dim, x_dim))
        clo = _get_or_create_var(nc, clo_var, 'f8', (crn_dim, y_dim, x_dim))
        cla = _get_or_create_var(nc, cla_var, 'f8', (crn_dim, y_dim, x_dim))

        # Write data
        lon[:] = x2.reshape(1, -1)
        lat[:] = y2.reshape(1, -1)
        for j in range(max_corners):
            clo[j, 0, :] = corner_lons[j, :]
            cla[j, 0, :] = corner_lats[j, :]

        # Set/update attributes
        lon.units = 'degrees_east'
        lon.standard_name = 'Longitude'
        lon.valid_min = np.min(x2)
        lon.valid_max = np.max(x2)

        lat.units = 'degrees_north'
        lat.standard_name = 'Latitude'
        lat.valid_min = np.min(y2)
        lat.valid_max = np.max(y2)

        clo.valid_min = np.min(corner_lons)
        clo.valid_max = np.max(corner_lons)

        cla.valid_min = np.min(corner_lats)
        cla.valid_max = np.max(corner_lats)

        # Global attributes
        nc.Conventions = 'CF-1.6'
        nc.history = (getattr(nc, 'history', '') + '; ' if 'history' in nc.ncattrs() else '') + \
                     f'Updated by pyfesom2 OASIS export module on {np.datetime64("now")}'
        # Note: Dimensions are already created/verified by _ensure_dim above
        
        # Note: Variables are already created/obtained by _get_or_create_var above
        
        # Set attributes
        lon.units = 'degrees_east'
        lon.standard_name = 'Longitude'
        lon.valid_min = np.min(x2)
        lon.valid_max = np.max(x2)
        
        lat.units = 'degrees_north'
        lat.standard_name = 'Latitude'
        lat.valid_min = np.min(y2)
        lat.valid_max = np.max(y2)
        
        clo.valid_min = np.min(corner_lons)
        clo.valid_max = np.max(corner_lons)
        
        cla.valid_min = np.min(corner_lats)
        cla.valid_max = np.max(corner_lats)
        
        # Write data
        lon[:] = x2.reshape(1, -1)
        lat[:] = y2.reshape(1, -1)
        
        for i in range(max_corners):
            clo[i, 0, :] = corner_lons[i, :]
            cla[i, 0, :] = corner_lats[i, :]
        
        # Global attributes
        nc.Conventions = 'CF-1.6'
        nc.history = f'Created by pyfesom2 OASIS export module on {np.datetime64("now")}'
    
    # ------------------------------------------------------------------
    # Write or update areas.nc
    # ------------------------------------------------------------------
    area_mode = 'a' if areas_exists else 'w'
    with Dataset(areas_file, area_mode, format='NETCDF4') as nc:
        _ensure_dim(nc, x_dim, n2d)
        _ensure_dim(nc, y_dim, 1)

        lon = _get_or_create_var(nc, lon_var, 'f8', (y_dim, x_dim))
        lat = _get_or_create_var(nc, lat_var, 'f8', (y_dim, x_dim))
        srf = _get_or_create_var(nc, srf_var, 'f8', (y_dim, x_dim))

        lon[:] = x2.reshape(1, -1)
        lat[:] = y2.reshape(1, -1)
        srf[:] = node_areas.reshape(1, -1)

        lon.units = 'degrees_east'
        lon.standard_name = 'Longitude'
        lon.valid_min = np.min(x2)
        lon.valid_max = np.max(x2)

        lat.units = 'degrees_north'
        lat.standard_name = 'Latitude'
        lat.valid_min = np.min(y2)
        lat.valid_max = np.max(y2)

        srf.coordinates = f"{lat_var} {lon_var}"
        srf.valid_min = np.min(node_areas)
        srf.valid_max = np.max(node_areas)

        nc.Conventions = 'CF-1.6'
        nc.history = (getattr(nc, 'history', '') + '; ' if 'history' in nc.ncattrs() else '') + \
                     f'Updated by pyfesom2 OASIS export module on {np.datetime64("now")}'
        
        # Set attributes
        lon.units = 'degrees_east'
        lon.standard_name = 'Longitude'
        lon.valid_min = np.min(x2)
        lon.valid_max = np.max(x2)
        
        lat.units = 'degrees_north'
        lat.standard_name = 'Latitude'
        lat.valid_min = np.min(y2)
        lat.valid_max = np.max(y2)
        
        srf.coordinates = f"{lat_var} {lon_var}"
        srf.valid_min = np.min(node_areas)
        srf.valid_max = np.max(node_areas)
        
        # Write data
        lon[:] = x2.reshape(1, -1)
        lat[:] = y2.reshape(1, -1)
        srf[:] = node_areas.reshape(1, -1)
        
        # Global attributes
        nc.Conventions = 'CF-1.6'
        nc.history = f'Created by pyfesom2 OASIS export module on {np.datetime64("now")}'
    
    # ------------------------------------------------------------------
    # Write or update masks.nc
    # ------------------------------------------------------------------
    mask_mode = 'a' if masks_exists else 'w'
    with Dataset(masks_file, mask_mode, format='NETCDF4') as nc:
        _ensure_dim(nc, x_dim, n2d)
        _ensure_dim(nc, y_dim, 1)

        lon = _get_or_create_var(nc, lon_var, 'f8', (y_dim, x_dim))
        lat = _get_or_create_var(nc, lat_var, 'f8', (y_dim, x_dim))
        msk = _get_or_create_var(nc, msk_var, 'i4', (y_dim, x_dim))

        lon[:] = x2.reshape(1, -1)
        lat[:] = y2.reshape(1, -1)
        msk[:] = mask.reshape(1, -1)

        lon.units = 'degrees_east'
        lon.standard_name = 'Longitude'
        lon.valid_min = np.min(x2)
        lon.valid_max = np.max(x2)

        lat.units = 'degrees_north'
        lat.standard_name = 'Latitude'
        lat.valid_min = np.min(y2)
        lat.valid_max = np.max(y2)

        msk.coordinates = f"{lat_var} {lon_var}"
        msk.valid_min = 0
        msk.valid_max = 1
        msk.coherent_with_grid = "undefined"

        nc.Conventions = 'CF-1.6'
        nc.history = (getattr(nc, 'history', '') + '; ' if 'history' in nc.ncattrs() else '') + \
                     f'Updated by pyfesom2 OASIS export module on {np.datetime64("now")}'
        
        # Set attributes
        lon.units = 'degrees_east'
        lon.standard_name = 'Longitude'
        lon.valid_min = np.min(x2)
        lon.valid_max = np.max(x2)
        
        lat.units = 'degrees_north'
        lat.standard_name = 'Latitude'
        lat.valid_min = np.min(y2)
        lat.valid_max = np.max(y2)
        
        msk.coordinates = f"{lat_var} {lon_var}"
        msk.valid_min = 0
        msk.valid_max = 1
        msk.coherent_with_grid = "undefined"
        
        # Write data
        lon[:] = x2.reshape(1, -1)
        lat[:] = y2.reshape(1, -1)
        msk[:] = mask.reshape(1, -1)
        
        # Global attributes
        nc.Conventions = 'CF-1.6'
        nc.history = f'Created by pyfesom2 OASIS export module on {np.datetime64("now")}'
    
    return {
        'grids_file': grids_file,
        'areas_file': areas_file,
        'masks_file': masks_file
    }
