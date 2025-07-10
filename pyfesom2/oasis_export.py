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
    
    # Check if files exist and handle overwrite
    for file_path in [grids_file, areas_file, masks_file]:
        if os.path.exists(file_path) and not overwrite:
            raise FileExistsError(f"File {file_path} already exists. Set overwrite=True to replace.")
    
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

    # Calculate element areas if not already in mesh
    if is_dict and 'elemareas' in mesh:
        voltri = mesh['elemareas']
    elif not is_dict and hasattr(mesh, 'voltri'):
        voltri = mesh.voltri
    else:
        # Simple spherical triangle area calculation
        voltri = np.zeros(e2d)
        R = 6371000.0  # Earth radius in meters
        rad = np.pi / 180.0
        
        for i in range(e2d):
            n1, n2, n3 = elem[i, 0] - 1, elem[i, 1] - 1, elem[i, 2] - 1
            
            # Convert to radians
            lon1, lat1 = x2[n1] * rad, y2[n1] * rad
            lon2, lat2 = x2[n2] * rad, y2[n2] * rad
            lon3, lat3 = x2[n3] * rad, y2[n3] * rad
            
            # Convert to Cartesian coordinates
            x1 = np.cos(lat1) * np.cos(lon1)
            y1 = np.cos(lat1) * np.sin(lon1)
            z1 = np.sin(lat1)
            
            x2_cart = np.cos(lat2) * np.cos(lon2)
            y2_cart = np.cos(lat2) * np.sin(lon2)
            z2 = np.sin(lat2)
            
            x3 = np.cos(lat3) * np.cos(lon3)
            y3 = np.cos(lat3) * np.sin(lon3)
            z3 = np.sin(lat3)
            
            # Calculate triangle area using cross product
            a = np.array([x2_cart - x1, y2_cart - y1, z2 - z1])
            b = np.array([x3 - x1, y3 - y1, z3 - z1])
            
            cross = np.cross(a, b)
            area = 0.5 * np.sqrt(np.sum(cross**2))
            
            # Convert to actual area on sphere
            voltri[i] = area * R**2
    
    # Calculate areas at nodes by distributing element areas
    node_areas = np.zeros(n2d)
    node_count = np.zeros(n2d)
    
    for i in range(e2d):
        n1, n2, n3 = elem[i, 0] - 1, elem[i, 1] - 1, elem[i, 2] - 1
        node_areas[n1] += voltri[i] / 3.0
        node_areas[n2] += voltri[i] / 3.0
        node_areas[n3] += voltri[i] / 3.0
        node_count[n1] += 1
        node_count[n2] += 1
        node_count[n3] += 1
    
    # Avoid division by zero
    node_count[node_count == 0] = 1
    
    # Create mask (1 for ocean, 0 for land)
    # In FESOM all nodes are considered wet (ocean)
    mask = np.ones(n2d, dtype=np.int32)
    
    # Create corner coordinates arrays
    # For each node, find connected elements and get the coordinates of the centroids
    node_corners = [[] for _ in range(n2d)]
    
    for i in range(e2d):
        n1, n2, n3 = elem[i, 0] - 1, elem[i, 1] - 1, elem[i, 2] - 1
        
        # Calculate element centroid
        x_cent = (x2[n1] + x2[n2] + x2[n3]) / 3.0
        y_cent = (y2[n1] + y2[n2] + y2[n3]) / 3.0
        
        # Add centroid to each node's corner list
        node_corners[n1].append((x_cent, y_cent))
        node_corners[n2].append((x_cent, y_cent))
        node_corners[n3].append((x_cent, y_cent))
    
    # For OASIS files, we need exactly 4 corners per node
    # For nodes with fewer than 4 connected elements, we'll duplicate the last corner
    # For nodes with more than 4, we'll select 4 corners that form a convex hull around the node
    max_corners = 4  # OASIS uses 4 corners
    corner_lons = np.zeros((max_corners, n2d))
    corner_lats = np.zeros((max_corners, n2d))
    
    for i in range(n2d):
        corners = node_corners[i]
        n_corners = len(corners)
        
        if n_corners == 0:
            # Node has no connected elements (should not happen in a valid mesh)
            # Use node coordinates for all corners as fallback
            for j in range(max_corners):
                corner_lons[j, i] = x2[i]
                corner_lats[j, i] = y2[i]
        elif n_corners <= max_corners:
            # Not enough corners, use what we have and duplicate the last one
            for j in range(n_corners):
                corner_lons[j, i] = corners[j][0]
                corner_lats[j, i] = corners[j][1]
            
            # Duplicate last corner if needed
            for j in range(n_corners, max_corners):
                corner_lons[j, i] = corners[-1][0]
                corner_lats[j, i] = corners[-1][1]
        else:
            # Too many corners, need to select 4
            # Simple approach: take corners at approximately equal intervals
            indices = np.linspace(0, n_corners-1, max_corners, dtype=int)
            for j, idx in enumerate(indices):
                corner_lons[j, i] = corners[idx][0]
                corner_lats[j, i] = corners[idx][1]

    # Create grids.nc file
    with Dataset(grids_file, 'w', format='NETCDF4') as nc:
        # Create dimensions
        nc.createDimension(x_dim, n2d)
        nc.createDimension(y_dim, 1)
        nc.createDimension(crn_dim, max_corners)
        
        # Create variables
        lon = nc.createVariable(lon_var, 'f8', (y_dim, x_dim))
        lat = nc.createVariable(lat_var, 'f8', (y_dim, x_dim))
        clo = nc.createVariable(clo_var, 'f8', (crn_dim, y_dim, x_dim))
        cla = nc.createVariable(cla_var, 'f8', (crn_dim, y_dim, x_dim))
        
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
    
    # Create areas.nc file
    with Dataset(areas_file, 'w', format='NETCDF4') as nc:
        # Create dimensions
        nc.createDimension(x_dim, n2d)
        nc.createDimension(y_dim, 1)
        
        # Create variables
        lon = nc.createVariable(lon_var, 'f8', (y_dim, x_dim))
        lat = nc.createVariable(lat_var, 'f8', (y_dim, x_dim))
        srf = nc.createVariable(srf_var, 'f8', (y_dim, x_dim))
        
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
    
    # Create masks.nc file
    with Dataset(masks_file, 'w', format='NETCDF4') as nc:
        # Create dimensions
        nc.createDimension(x_dim, n2d)
        nc.createDimension(y_dim, 1)
        
        # Create variables
        lon = nc.createVariable(lon_var, 'f8', (y_dim, x_dim))
        lat = nc.createVariable(lat_var, 'f8', (y_dim, x_dim))
        msk = nc.createVariable(msk_var, 'i4', (y_dim, x_dim))
        
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
