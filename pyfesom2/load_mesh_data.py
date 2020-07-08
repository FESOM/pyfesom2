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


def load_mesh(path, abg=[0, 0, 0], usepickle=True, usejoblib=False, protocol=4):
    """ Loads FESOM mesh

    Parameters
    ----------
    path : str
        Path to the directory with mesh files
    abg : list
        alpha, beta and gamma Euler angles. Rotated meshes use [50, 15, -90]
    usepickle: bool
        use pickle file to store or load mesh data
    usejoblib: bool
        use joblib file to store or load mesh data
    protocol: int
        used for pickle, only way to save data more than 4 Gb
    Returns
    -------
    mesh : object
        fesom_mesh object
    """
    path = os.path.abspath(path)
    if (usepickle == True) and (usejoblib == True):
        raise ValueError(
            "Both `usepickle` and `usejoblib` set to True, select only one"
        )

    if usepickle:
        pickle_file = os.path.join(path, "pickle_mesh_py3_fesom2")
        print(pickle_file)

    if usejoblib:
        joblib_file = os.path.join(path, "joblib_mesh_fesom2")

    if usepickle and (os.path.isfile(pickle_file)):
        print("The usepickle == True)")
        print("The pickle file for FESOM2 exists.")
        print("The mesh will be loaded from {}".format(pickle_file))

        ifile = open(pickle_file, "rb")
        mesh = pickle.load(ifile)
        ifile.close()
        return mesh

    elif (usepickle == True) and (os.path.isfile(pickle_file) == False):
        print("The usepickle == True")
        print("The pickle file for FESOM2 DO NOT exists")
        print("The mesh will be saved to {}".format(pickle_file))

        mesh = fesom_mesh(path=path, abg=abg)
        logging.info("Use pickle to save the mesh information")
        print("Save mesh to binary format")
        outfile = open(pickle_file, "wb")
        pickle.dump(mesh, outfile, protocol=protocol)
        outfile.close()
        return mesh

    elif (usepickle == False) and (usejoblib == False):
        mesh = fesom_mesh(path=path, abg=abg)
        return mesh

    if (usejoblib == True) and (os.path.isfile(joblib_file)):
        print("The usejoblib == True)")
        print("The joblib file for FESOM2 exists.")
        print("The mesh will be loaded from {}".format(joblib_file))

        mesh = joblib.load(joblib_file)
        return mesh

    elif (usejoblib == True) and (os.path.isfile(joblib_file) == False):
        print("The usejoblib == True")
        print("The joblib file for FESOM2 DO NOT exists")
        print("The mesh will be saved to {}".format(joblib_file))

        mesh = fesom_mesh(path=path, abg=abg)
        logging.info("Use joblib to save the mesh information")
        print("Save mesh to binary format")
        joblib.dump(mesh, joblib_file)

        return mesh


class fesom_mesh(object):
    """ Creates instance of the FESOM mesh.
    This class creates instance that contain information
    about FESOM mesh. At present the class works with
    ASCII representation of the FESOM grid, but should be extended
    to be able to read also netCDF version (probably UGRID convention).

    Minimum requirement is to provide the path to the directory,
    where following files should be located (not nessesarely all of them will
    be used):

    - nod2d.out
    - elem2d.out
    - aux3d.out

    Parameters
    ----------
    path : str
        Path to the directory with mesh files

    abg : list
        alpha, beta and gamma Euler angles. Rotated meshes use [50, 15, -90]


    Attributes
    ----------
    path : str
        Path to the directory with mesh files
    x2 : array
        x position (lon) of the surface node
    y2 : array
        y position (lat) of the surface node
    n2d : int
        number of 2d nodes
    e2d : int
        number of 2d elements (triangles)
    nlev : int
        number of vertical levels
    zlevs : array
        array of vertical level depths
    voltri

    existing instances are: path, n2d, e2d,
    nlev, zlevs, x2, y2, elem, no_cyclic_elem, alpha, beta, gamma

    Returns
    -------
    mesh : object
        fesom_mesh object
    """

    def __init__(self, path, abg=[50, 15, -90]):
        self.path = os.path.abspath(path)

        if not os.path.exists(self.path):
            raise IOError('The path "{}" does not exists'.format(self.path))

        self.alpha = abg[0]
        self.beta = abg[1]
        self.gamma = abg[2]

        self.nod2dfile = os.path.join(self.path, "nod2d.out")
        self.elm2dfile = os.path.join(self.path, "elem2d.out")
        self.aux3dfile = os.path.join(self.path, "aux3d.out")

        self.e2d = 0
        self.nlev = 0
        self.zlevs = []
        self.topo = []
        self.voltri = []

        logging.info("load 2d part of the mesh")
        if (sys.version_info.major, sys.version_info.minor) >= (3, 7):
            start = time.time()
        else:
            start = time.clock()
        self.read2d()
        if (sys.version_info.major, sys.version_info.minor) >= (3, 7):
            end = time.time()
        else:
            end = time.clock()
        print("Load 2d part of the mesh in {} second(s)".format(str(int(end - start))))

    def read2d(self):
        file_content = pd.read_csv(
            self.nod2dfile,
            delim_whitespace=True,
            skiprows=1,
            names=["node_number", "x", "y", "flag"],
        )
        self.x2 = file_content.x.values
        self.y2 = file_content.y.values
        self.ind2d = file_content.flag.values
        self.n2d = len(self.x2)

        file_content = pd.read_csv(
            self.elm2dfile,
            delim_whitespace=True,
            skiprows=1,
            names=["first_elem", "second_elem", "third_elem"],
        )
        self.elem = file_content.values - 1
        self.e2d = np.shape(self.elem)[0]

        ###########################################
        # here we compute the volumes of the triangles
        # this should be moved into fesom general mesh output netcdf file
        #
        r_earth = 6371000.0
        rad = np.pi / 180
        edx = self.x2[self.elem]
        edy = self.y2[self.elem]
        ed = np.array([edx, edy])

        jacobian2D = ed[:, :, 1] - ed[:, :, 0]
        jacobian2D = np.array([jacobian2D, ed[:, :, 2] - ed[:, :, 0]])
        for j in range(2):
            mind = [i for (i, val) in enumerate(jacobian2D[j, 0, :]) if val > 355]
            pind = [i for (i, val) in enumerate(jacobian2D[j, 0, :]) if val < -355]
            jacobian2D[j, 0, mind] = jacobian2D[j, 0, mind] - 360
            jacobian2D[j, 0, pind] = jacobian2D[j, 0, pind] + 360

        jacobian2D = jacobian2D * r_earth * rad

        for k in range(2):
            jacobian2D[k, 0, :] = jacobian2D[k, 0, :] * np.cos(edy * rad).mean(axis=1)

        self.voltri = abs(np.linalg.det(np.rollaxis(jacobian2D, 2))) / 2.0

        # compute the 2D lump operator

        self.lump2 = np.array((0.0,) * self.n2d)
        for i in range(3):
            for j in range(self.e2d):
                n = self.elem[j, i]
                self.lump2[n] = self.lump2[n] + self.voltri[j]
        self.lump2 = self.lump2 / 3.0

        self.x2, self.y2 = scalar_r2g(
            self.alpha, self.beta, self.gamma, self.x2, self.y2
        )
        d = self.x2[self.elem].max(axis=1) - self.x2[self.elem].min(axis=1)
        self.no_cyclic_elem = [i for (i, val) in enumerate(d) if val < 100]

        with open(self.aux3dfile) as f:
            self.nlev = int(next(f))
            self.zlev = np.array([next(f).rstrip() for x in range(self.nlev)]).astype(
                float
            )

        topo_pd = pd.read_csv(self.aux3dfile, skiprows=self.nlev)
        self.topo = topo_pd.values[:, 0]

        return self

    def meshinfo(self):
        self.meshinfo_text = """
FESOM mesh:
path                  = {}
alpha, beta, gamma    = {}, {}, {}
number of 2d nodes    = {}
number of 2d elements = {}

        """.format(
            self.path,
            str(self.alpha),
            str(self.beta),
            str(self.gamma),
            str(self.n2d),
            str(self.e2d),
        )
        return self.meshinfo_text

    def __repr__(self):
        return self.meshinfo()

    def __str__(self):
        return self.meshinfo()


def ind_for_depth(depth, mesh):
    """
    Find the model depth index that is closest to the required depth

    Parameters
    ----------
    depth : float
        desired depth.
    mesh : object
        FESOM mesh object

    Returns
    dind : int
        index that corresponds to the model depth level closest to `depth`.
    -------
    """
    arr = [abs(abs(z) - abs(depth)) for z in mesh.zlev]
    v, i = min((v, i) for (i, v) in enumerate(arr))
    dind = i
    return dind


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

    if ("nz1" in dataset.dims) and (depth != None):
        if (continuous == True) and len(data.dims == 3) and (dind.step == None):
            data = data[:, :, dind]
        elif (continuous == True) and len(data.dims == 2) and (dind.step == None):
            data = data[:, dind]
        elif continuous == False:
            data = data.isel(nz1=dind)
    elif ("nz" in dataset.dims) and (depth != None):
        if (continuous == True) and len(data.dims == 3) and (dind.step == None):
            data = data[:, :, dind]
        elif (continuous == True) and len(data.dims == 2) and (dind.step == None):
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
    silent = False,
    **kwargs
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
            fname = "{}.{}.{}.nc".format(variable, runid, year)
            path = os.path.join(result_path, fname)
            paths.append(path)
    elif isinstance(years, int):
        fname = "{}.{}.{}.nc".format(variable, runid, years)
        paths = os.path.join(result_path, fname)
    else:
        raise ValueError("year can be integer, list or one dimentional numpy array")

    if depth != None:
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

    if compute:
        data = data.compute()
        data = data.data

    #     median min sum std var
    return data
