# -*- coding: utf-8 -*-
#
# This file is part of pyfesom2
# Original code by Dmitry Sidorenko, Nikolay Koldunov,
# Qiang Wang, Sergey Danilov and Patrick Scholz
#

import logging
import os
import pickle
import sys
import time

import joblib
import numpy as np
import pandas as pd
import pyresample
import xarray as xr
from netCDF4 import Dataset

from .ut import scalar_r2g
from typing import List, Tuple, Optional


def load_mesh(path, abg=[0, 0, 0], cache_method=None, protocol=4):
    """Loads FESOM mesh with efficient caching

    Parameters
    ----------
    path : str
        Path to the directory with mesh files
    abg : list
        alpha, beta and gamma Euler angles. Rotated meshes use [50, 15, -90]
    cache_method : str or None
        Method to cache mesh data: 'pickle', 'joblib', or None for no caching
    protocol : int
        Protocol used for pickle serialization, required for data > 4 GB

    Returns
    -------
    mesh : object
        fesom_mesh object
    """
    
    # Conditional import to avoid dependency if joblib isn't used
    if cache_method == 'joblib':
        try:
            import joblib
        except ImportError:
            logging.warning("joblib not available; falling back to pickle")
            cache_method = 'pickle'
    
    path = os.path.abspath(path)
    
    # Set up cache directories
    mesh_name = os.path.basename(path)
    cache_dir = os.environ.get("PYFESOM_CACHE", os.path.join(os.getcwd(), "MESH_cache"))
    cache_dir = os.path.join(cache_dir, mesh_name)
    
    # Initialize cache file paths
    cache_file = None
    
    if cache_method:
        # Create cache directory if it doesn't exist
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
            
        # Set up cache filenames based on method
        filename = f"{'pickle' if cache_method == 'pickle' else 'joblib'}_mesh_py3_fesom2"
        
        # Check possible locations for cache file
        if os.path.isfile(os.path.join(path, filename)):
            cache_file = os.path.join(path, filename)
            logging.info(f"Using cached mesh from {cache_file}")
        elif os.path.isfile(os.path.join(cache_dir, filename)):
            cache_file = os.path.join(cache_dir, filename)
            logging.info(f"Using cached mesh from {cache_file}")
        else:
            logging.info(f"No cache file found, will create one with {cache_method}")
    
    # Try to load from cache if available
    if cache_method and cache_file and os.path.isfile(cache_file):
        try:
            if cache_method == 'pickle':
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            elif cache_method == 'joblib':
                return joblib.load(cache_file)
        except Exception as e:
            logging.warning(f"Failed to load cached mesh: {str(e)}")
            # Continue to load from raw files if cache loading fails
    
    # Create fresh mesh object if no cache or loading failed
    # from fesom_mesh import fesom_mesh  # Import here to avoid circular imports
    mesh = FesomMesh(path=path, abg=abg)
    
    # Try to save mesh to cache if requested
    if cache_method:
        # Try to save in mesh directory first, then in cache directory
        for save_dir in [path, cache_dir]:
            cache_file = os.path.join(save_dir, filename)
            try:
                if cache_method == 'pickle':
                    with open(cache_file, "wb") as f:
                        pickle.dump(mesh, f, protocol=protocol)
                elif cache_method == 'joblib':
                    joblib.dump(mesh, cache_file)
                
                logging.info(f"Mesh saved to {cache_file}")
                break  # Stop trying if successful
            except PermissionError:
                logging.warning(f"Permission denied saving to {cache_file}")
            except Exception as e:
                logging.warning(f"Failed to save mesh cache: {str(e)}")
    
    return mesh




class FesomMesh:
    """FESOM mesh representation.
    
    This class loads and stores information about the FESOM (Finite Element Sea Ice-Ocean Model) mesh.
    Currently works with ASCII representation of the FESOM grid, with plans to extend
    to netCDF format (UGRID convention).

    Parameters
    ----------
    path : str
        Path to the directory containing mesh files
    abg : list, optional
        Alpha, beta and gamma Euler angles. Default is [50, 15, -90] for rotated meshes.
    lazy_load : bool, optional
        If True, only load basic information on initialization and defer 2D mesh loading
        until explicitly requested. Default is False.

    Attributes
    ----------
    path : str
        Path to the directory with mesh files
    x2 : ndarray
        Longitude positions of the surface nodes
    y2 : ndarray
        Latitude positions of the surface nodes
    ind2d : ndarray
        Flag values for 2D nodes
    n2d : int
        Number of 2D nodes
    elem : ndarray
        Element connectivity array
    e2d : int
        Number of 2D elements (triangles)
    nlev : int
        Number of vertical levels
    zlev : ndarray
        Array of vertical level depths
    topo : ndarray
        Topography information
    voltri : ndarray
        Volume of triangular elements
    lump2 : ndarray
        2D lumped operator values
    no_cyclic_elem : ndarray
        Indices of non-cyclic elements
    alpha, beta, gamma : float
        Euler angles for coordinate transformation
        
    Examples
    --------
    >>> mesh = FesomMesh('/path/to/mesh/files')
    >>> print(mesh)
    >>> # Access node coordinates
    >>> lons, lats = mesh.x2, mesh.y2
    """

    def __init__(self, path: str, abg: List[float] = [50, 15, -90], lazy_load: bool = False):
        """Initialize the FESOM mesh object."""
        self.path = os.path.abspath(path)
        if not os.path.exists(self.path):
            raise IOError(f'The path "{self.path}" does not exist')

        # Set Euler angles for coordinate transformation
        self.alpha, self.beta, self.gamma = abg
        
        # Set file paths
        self.nod2dfile = os.path.join(self.path, "nod2d.out")
        self.elm2dfile = os.path.join(self.path, "elem2d.out")
        self.aux3dfile = os.path.join(self.path, "aux3d.out")
        
        # Check if required files exist
        self._check_files_exist()
        
        # Initialize attributes
        self.x2 = None
        self.y2 = None
        self.ind2d = None
        self.n2d = 0
        self.elem = None
        self.e2d = 0
        self.nlev = 0
        self.zlev = None
        self.topo = None
        self.voltri = None
        self.lump2 = None
        self.no_cyclic_elem = None
        
        # Load vertical structure from aux3d.out immediately
        self._load_vertical_structure()
        
        # Load 2D mesh immediately unless lazy loading is requested
        if not lazy_load:
            self.load_2d_mesh()
            
    def _check_files_exist(self) -> None:
        """Verify that required mesh files exist."""
        required_files = [self.nod2dfile, self.elm2dfile, self.aux3dfile]
        missing_files = [f for f in required_files if not os.path.isfile(f)]
        
        if missing_files:
            raise FileNotFoundError(f"Missing required mesh files: {', '.join(missing_files)}")
    
    def _load_vertical_structure(self) -> None:
        """Load vertical level information from aux3d file."""
        try:
            with open(self.aux3dfile) as f:
                self.nlev = int(next(f))
                self.zlev = np.array([next(f).rstrip() for _ in range(self.nlev)]).astype(float)
                
            # Load topography data after vertical levels
            topo_pd = pd.read_csv(self.aux3dfile, skiprows=self.nlev)
            self.topo = topo_pd.values[:, 0]
        except Exception as e:
            logging.error(f"Failed to load vertical structure: {str(e)}")
            raise

    def load_2d_mesh(self) -> 'FesomMesh':
        """Load 2D part of the mesh.
        
        Returns
        -------
        self : FesomMesh
            Returns self for method chaining
        """
        logging.info("Loading 2D part of the mesh...")
        
        # Use time.perf_counter() for Python 3.3+ as it's more accurate
        # and works across all platforms
        timer_func = time.perf_counter if hasattr(time, 'perf_counter') else time.clock
        start_time = timer_func()
        
        try:
            # Load node coordinates
            node_data = pd.read_csv(
                self.nod2dfile,
                sep=r'\s+',
                skiprows=1,
                names=["node_number", "x", "y", "flag"],
            )
            self.x2 = node_data.x.values
            self.y2 = node_data.y.values
            self.ind2d = node_data.flag.values
            self.n2d = len(self.x2)
            
            # Load element connectivity
            elem_data = pd.read_csv(
                self.elm2dfile,
                sep=r'\s+',
                skiprows=1,
                names=["first_elem", "second_elem", "third_elem"],
            )
            self.elem = elem_data.values - 1  # Convert to 0-based indexing
            self.e2d = self.elem.shape[0]
            
            # Compute triangle volumes
            self._compute_triangle_volumes()
            
            # Apply coordinate transformation
            self._transform_coordinates()
            
            # Identify non-cyclic elements
            self._identify_noncyclic_elements()
            
            end_time = timer_func()
            logging.info(f"Loaded 2D mesh in {int(end_time - start_time)} second(s)")
            
            return self
            
        except Exception as e:
            logging.error(f"Failed to load 2D mesh: {str(e)}")
            raise
    
    def _compute_triangle_volumes(self) -> None:
        """Compute volumes of triangular elements."""
        r_earth = 6371000.0  # Earth radius in meters
        rad = np.pi / 180    # Degree to radian conversion
        
        # Extract coordinates of triangle vertices
        edx = self.x2[self.elem]
        edy = self.y2[self.elem]
        
        # Initialize Jacobian for the transformation
        jacobian2D = np.zeros((2, 2, self.e2d))
        jacobian2D[0] = edx[:, 1] - edx[:, 0], edx[:, 2] - edx[:, 0]
        jacobian2D[1] = edy[:, 1] - edy[:, 0], edy[:, 2] - edy[:, 0]
        
        # Handle coordinate wrapping at the dateline
        for j in range(2):
            # Find indices where the difference is larger than half the globe
            for i in range(2):
                # Where difference > 355, subtract 360
                mask = jacobian2D[j, i, :] > 355
                jacobian2D[j, i, mask] -= 360
                
                # Where difference < -355, add 360
                mask = jacobian2D[j, i, :] < -355
                jacobian2D[j, i, mask] += 360
        
        # Convert to meters
        jacobian2D = jacobian2D * r_earth * rad
        
        # Adjust for latitude (cos(lat) factor)
        for k in range(2):
            jacobian2D[k, :, :] *= np.cos(np.radians(edy)).mean(axis=1)
        
        # Calculate area using determinant
        self.voltri = abs(np.linalg.det(np.rollaxis(jacobian2D, 2))) / 2.0
        
        # Compute the 2D lumped operator
        self.lump2 = np.zeros(self.n2d)
        for i in range(3):
            for j in range(self.e2d):
                n = self.elem[j, i]
                self.lump2[n] += self.voltri[j]
        self.lump2 /= 3.0
    
    def _transform_coordinates(self) -> None:
        """Transform coordinates using Euler angles."""
        from .ut import scalar_r2g  # Import here to avoid circular imports
        
        self.x2, self.y2 = scalar_r2g(
            self.alpha, self.beta, self.gamma, self.x2, self.y2
        )
    
    def _identify_noncyclic_elements(self) -> None:
        """Identify elements that don't cross the dateline."""
        # Calculate max longitude difference within each element
        lon_range = self.x2[self.elem].max(axis=1) - self.x2[self.elem].min(axis=1)
        
        # Elements with longitude range < 100 degrees are considered non-cyclic
        self.no_cyclic_elem = np.where(lon_range < 100)[0]
    
    def get_element_area(self, elem_indices=None) -> np.ndarray:
        """Get areas of specified triangular elements.
        
        Parameters
        ----------
        elem_indices : array-like, optional
            Indices of elements to get areas for. If None, returns areas for all elements.
            
        Returns
        -------
        ndarray
            Areas of the specified elements in square meters
        """
        if self.voltri is None:
            raise RuntimeError("Mesh elements not loaded. Call load_2d_mesh() first.")
            
        if elem_indices is None:
            return self.voltri
        return self.voltri[elem_indices]
    
    def get_mesh_stats(self) -> dict:
        """Get basic statistics about the mesh.
        
        Returns
        -------
        dict
            Dictionary containing mesh statistics
        """
        stats = {
            'n2d': self.n2d,
            'e2d': self.e2d,
            'nlev': self.nlev,
            'min_depth': self.zlev.min() if self.zlev is not None else None,
            'max_depth': self.zlev.max() if self.zlev is not None else None,
        }
        
        if self.voltri is not None:
            stats.update({
                'min_elem_area': self.voltri.min(),
                'max_elem_area': self.voltri.max(),
                'mean_elem_area': self.voltri.mean(),
            })
            
        return stats

    def meshinfo(self) -> str:
        """Get formatted mesh information.
        
        Returns
        -------
        str
            Formatted information about the mesh
        """
        info = f"""
FESOM mesh:
path                  = {self.path}
alpha, beta, gamma    = {self.alpha}, {self.beta}, {self.gamma}
number of 2d nodes    = {self.n2d}
number of 2d elements = {self.e2d}
number of levels      = {self.nlev}
"""
        if self.voltri is not None:
            info += f"total surface area    = {self.voltri.sum():e} m²\n"
            
        return info

    def __repr__(self) -> str:
        """Return string representation of the mesh."""
        return self.meshinfo()

    def __str__(self) -> str:
        """Return string representation of the mesh."""
        return self.meshinfo()


def ind_for_depth(depth: float, mesh) -> int:
    """
    Find the model depth index that is closest to the required depth.
    
    Uses vectorized NumPy operations for better performance.
    
    Parameters
    ----------
    depth : float
        Desired depth (positive or negative).
    mesh : object
        FESOM mesh object with zlev attribute.
        
    Returns
    -------
    int
        Index of the model depth level closest to the requested `depth`.
        
    Examples
    --------
    >>> mesh = load_mesh('/path/to/mesh')
    >>> # Find index closest to 100m depth
    >>> idx = ind_for_depth(100, mesh)
    >>> print(f"Closest depth level is {mesh.zlev[idx]} m")
    """
    # Handle input validation
    if mesh.zlev is None or len(mesh.zlev) == 0:
        raise ValueError("Mesh object does not have valid vertical levels")
    
    # Convert to numpy array if not already
    zlev = np.asarray(mesh.zlev)
    
    # Convert all depths to absolute values and find closest match
    # This handles both positive and negative depth values
    distance = np.abs(np.abs(zlev) - np.abs(depth))
    
    # Return the index of the minimum distance
    return np.argmin(distance)


def select_slices(dataset, variable, mesh, records, depth, continuous=False):
    """Select slices from data.

    The xarray isel function can be very slow, so in order to use arbitrary
    selection, we add an option to acces the data directly by index.

    To use it the `continuous` argument should be True. That mean you
    provide as record a slice with step 1. In other cases you should rely on
    xarray isel.

    Parameters
    ----------
    dataset: xarray.DataSet
        input data
    variable: str
        name of the variable
    mesh: mesh object
        FESOM2 mesh object.
    records: int, slice, list
        number of time steps to be considered for aggregation.
        If -1 (default), all timesteps will be taken in to account.
        If 0, only the first record will be taken
        If [0,5,7], only time steps with indexes 0,5 and 7 will be taken
        If slice(2,120,12), every 12th time step starting from the third one will be selected.
    depth: float
        The model depth closest to provided depth will be taken.
        If None, 3d field will be returned. Default = None.
    continuous: bool
        If True the time steps will be selected by directly accesing indexes,
            this only possible if records is a slice with step 1
        If False, data will be selected with xarray isel

    Returns
    -------
    data: xarray.DataArray
        data selected over time and depth.
    """

    if depth != None:
        dind = ind_for_depth(depth, mesh)

    data = None
    if records == -1:
        data = dataset[variable]
    elif isinstance(records, slice):
        if (records.step == None) and (continuous == True):
            data = dataset[variable][records, :]
        elif (records.step != None) and (continuous == False):
            data = dataset[variable].isel(time=records)
        elif (records.step == None) and (continuous == False):
            data = dataset[variable].isel(time=records)
        elif (records.step != None) and (continuous == True):
            raise ValueError(
                "You set `continuous` to True, but the step in the slice is not None."
            )
    # lists are not allowed if continuous == True
    elif isinstance(records, list):
        if continuous == True:
            raise ValueError("You set `continuous` to True, lists are not allowed.")
        else:
            data = dataset[variable].isel(time=records)
    else:
        raise ValueError("Records should be ether -1 or instance of a list or a slice.")

        # this just can't work and probably never worked :)
    if ("nz1" in dataset.dims) and (depth != None):
        if (continuous == True) and (len(data.dims) == 3) and (dind.step == None):
            data = data[:, :, dind]
        elif (continuous == True) and (len(data.dims) == 2) and (dind.step == None):
            data = data[:, dind]
        elif continuous == False:
            data = data.isel(nz1=dind)
    elif ("nz" in dataset.dims) and (depth != None):
        if (continuous == True) and (len(data.dims) == 3) and (dind.step == None):
            data = data[:, :, dind]
        elif (continuous == True) and (len(data.dims) == 2) and (dind.step == None):
            data = data[:, dind]
        elif continuous == False:
            data = data.isel(nz=dind)
    return data


def get_data(
    result_path,
    variable,
    years,
    mesh,
    runid="fesom",
    records=-1,
    depth=None,
    how="mean",
    ncfile=None,
    compute=True,
    continuous=False,
    silent=False,
    transpose=True,
    naming_convention="fesom",
    naming_template=None,
    **kwargs,
):
    """
    Get the data at some depth level, agregated if needed.

    Parameters
    ----------
    result_path : string
        path to the data folder.
    variable : string
        variable name
    years : int, list
        year or list of years to open
    mesh: mesh object
        FESOM2 mesh object.
    records: int, slice, list
        number of time steps to be considered for aggregation.
        If -1 (default), all timesteps will be taken in to account.
        If 0, only the first record will be taken
        If [0,5,7], only time steps with indexes 0,5 and 7 will be taken
        If slice(2,120,12), every 12th time step starting from the third one will be selected.
    depth: float
        The model depth closest to provided depth will be taken.
        If None, 3d field will be returned. Default = None.
    how: str
        method of aggregation.
        Can be "mean" (default), "min", "max", "median", "min", "sum", "std", "var"
        If None, no aggregation is applied.
    ncfile: str
        if provided, the netCDF file will be opened directly.
        Some dummy data have to be provided for result_path and years
    compute: bool
        Do the actual computations or not. Default True.
    trnspose: bool
        Transpose to the old FESOM2.1 format with [time, nod2, nz1] dimension order.
    runid: str
        For esm-tools naming convention, use the experiment id name (e.g. test, PI-CTRL, LGM, ...)
    naming_convention : str
        The naming convention to be used. Can either be "fesom" for classic
        infrastructure, "esm-tools" for esm-tools infrastructure, or "custom",
        in which case a template string must be provided.
    naming_template : None or str
        Required if a customized naming convention is to be used. Replaced variables will be (in order) variable, runid, year.
    **kwargs: dict
       you can add aditional arguments to pass to the xarray.open_mfdataset (for example slice sizes)

    Returns
    -------
    data: xarray
        aggregated data at some depth level.

    """
    # if records == 0:
    #     records = 1

    paths = []
    if ncfile:
        paths = ncfile
    elif isinstance(years, (list, np.ndarray, range)):
        paths = []
        for year in years:
            if naming_convention == "fesom":
                fname = "{}.{}.{}.nc".format(variable, runid, year)
            elif naming_convention == "esm_tools":
                # FIXME(PG): Only for yearly restarts for now...
                fname = f"{runid}.{year}.{variable}01.01.nc"
            elif naming_convention == "custom":
                fname = naming_template.format(variable, runid, year)
            else:
                raise ValueError(
                    "You must have fesom, esm_tools, or custom as naming_convention!"
                )
            path = os.path.join(result_path, fname)
            paths.append(path)
    elif isinstance(years, int):
        if naming_convention == "fesom":
            fname = "{}.{}.{}.nc".format(variable, runid, years)
        elif naming_convention == "esm_tools":
            # FIXME(PG): Only for yearly restarts for now...
            fname = f"{runid}.{years}.{variable}01.01.nc"
        elif naming_convention == "custom":
            fname = naming_template.format(variable, runid, years)
        else:
            raise ValueError(
                "You must have fesom, esm_tools, or custom as naming_convention!"
            )
        paths = os.path.join(result_path, fname)
    else:
        raise ValueError("year can be integer, list or one dimentional numpy array")

    if depth is not None:
        dind = ind_for_depth(depth, mesh)
        if not silent:
            print("Model depth: {}".format(abs(mesh.zlev[dind])))
    else:
        if not silent:
            print("Depth is None, 3d field will be returned")

    dataset = xr.open_mfdataset(paths, combine="by_coords", **kwargs)
    data = select_slices(dataset, variable, mesh, records, depth)

    if how == "mean":
        data = data.mean(dim="time")
    elif how == "max":
        data = data.max(dim="time")
    elif how == "min":
        data = data.min(dim="time")
    elif how == "median":
        data = data.median(dim="time")
    elif how == "sum":
        data = data.sum(dim="time")
    elif how == "std":
        data = data.std(dim="time")
    elif how == "var":
        data = data.var(dim="time")
    elif how == "original":
        data = data
    else:
        pass

    if transpose:
        if len(data.dims) == 3:
            if (
                ("nz1" in data.dims)
                and ("nod2" in data.dims)
                and (data.dims != ("time", "nod2", "nz1"))
            ):
                data = data.transpose("time", "nod2", "nz1")
            elif (
                ("nz" in data.dims)
                and ("nod2" in data.dims)
                and (data.dims != ("time", "nod2", "nz"))
            ):
                data = data.transpose("time", "nod2", "nz")

            elif (
                ("nz1" in data.dims)
                and ("elem" in data.dims)
                and (data.dims != ("time", "elem", "nz1"))
            ):
                data = data.transpose("time", "elem", "nz1")
            elif (
                ("nz" in data.dims)
                and ("elem" in data.dims)
                and (data.dims != ("time", "elem", "nz"))
            ):
                data = data.transpose("time", "elem", "nz")

        elif len(data.dims) == 2:
            if (
                ("nz1" in data.dims)
                and ("nod2" in data.dims)
                and (data.dims != ("nod2", "nz1"))
            ):
                data = data.transpose("nod2", "nz1")
            elif (
                ("nz" in data.dims)
                and ("nod2" in data.dims)
                and (data.dims != ("nod2", "nz"))
            ):
                data = data.transpose("nod2", "nz")

            if (
                ("nz1" in data.dims)
                and ("elem" in data.dims)
                and (data.dims != ("elem", "nz1"))
            ):
                data = data.transpose("elem", "nz1")
            elif (
                ("nz" in data.dims)
                and ("elem" in data.dims)
                and (data.dims != ("elem", "nz"))
            ):
                data = data.transpose("elem", "nz")

    if compute:
        data = data.compute()
        data = data.data

    #     median min sum std var
    return data
