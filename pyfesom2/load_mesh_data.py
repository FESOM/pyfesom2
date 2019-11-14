# -*- coding: utf-8 -*-
#
# This file is part of pyfesom2
# Original code by Dmitry Sidorenko, 2013
#

import pandas as pd
import numpy as np
from netCDF4 import Dataset
from .ut import scalar_r2g
import os
import logging
import time
import pickle
import pyresample
import joblib
import xarray as xr


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
        start = time.clock()
        self.read2d()
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


def read_fesom_slice(
    str_id,
    records,
    year,
    mesh,
    result_path,
    runid,
    ilev=0,
    how="mean",
    verbose=False,
    ncfile="",
):
    # print(['reading year '+str(year)+':'])
    if ncfile == "":
        ncfile = result_path + "/" + str_id + "." + runid + "." + str(year) + ".nc"
    if verbose:
        print(["reading ", ncfile])
    f = Dataset(ncfile, "r")
    # dimensions of the netcdf variable
    ncdims = f.variables[str_id].shape
    # indexies for reading 2D part
    if verbose:
        if ncdims[1] == mesh.n2d:
            print("data at nodes")
        elif ncdims[1] == mesh.e2d:
            print("data on elements")
        else:
            raise IOError("not existing dimension " + str(ncdims[1]))

    dim = [records, np.arange(ncdims[1])]
    data = np.zeros(shape=(ncdims[1]))
    # add 3rd index if reading a slice from 3D data
    if len(ncdims) == 3:
        dim.append(ilev)

    if how == "mean":
        data = data + f.variables[str_id][dim].mean(axis=0)
    elif how == "max":
        data = data + f.variables[str_id][dim].max(axis=0)
    elif how == "min":
        data = data + f.variables[str_id][dim].min(axis=0)
    f.close()
    return data


def read_fesom_sect(
    str_id,
    records,
    year,
    mesh,
    result_path,
    runid,
    p1,
    p2,
    N,
    nlev=0,
    how="mean",
    line_distance=3.0,
    radius_of_influence=2000000,
    do_land=False,
    verbose=False,
):
    # define section: sx, sy, sz
    sx = np.linspace(p1[0], p2[0], N)
    sy = np.linspace(p1[1], p2[1], N)
    if nlev == 0:
        nlev = mesh.nlev
    sz = np.zeros([nlev, N])
    sz[:, :] = np.nan
    # find 2D points which are within the line_distance to the section
    # to make it cheaper, only these points will be used for interpolation
    lnorm = np.sqrt(sum((p2 - p1) ** 2))
    d = np.cross(np.array([mesh.x2, mesh.y2]).T - p1, p2 - p1) / lnorm
    ind = d < line_distance
    x = mesh.x2[ind]
    y = mesh.y2[ind]
    oce_ind2d = np.ones(x.shape)
    oce_ind2d[mesh.ind2d[ind] != 0] = np.nan
    # prepare the interpolation weights
    orig_def = pyresample.geometry.SwathDefinition(lons=x, lats=y)
    targ_def = pyresample.geometry.SwathDefinition(lons=sx, lats=sy)

    oce_mask = pyresample.kd_tree.resample_nearest(
        orig_def,
        oce_ind2d,
        targ_def,
        radius_of_influence=radius_of_influence,
        fill_value=0.0,
    )
    # do the interpolation layerwise
    for ilev in range(nlev):
        # read the model result from fesom.XXXX.oce.nc
        if verbose:
            print("interpolating level ", ilev)

        data = read_fesom_slice(
            str_id, records, year, mesh, result_path, runid, ilev=ilev
        )
        data[data == 0.0] = np.nan
        sz[ilev, :] = (
            pyresample.kd_tree.resample_gauss(
                orig_def,
                data[ind],
                targ_def,
                radius_of_influence=radius_of_influence,
                neighbours=10,
                sigmas=250000,
                fill_value=None,
            )
            * oce_mask
        )

    return (sx, sy, sz)


def cut_region(mesh, nlevels, box=[13, 30, 53, 66], depth=0):
    """
    Cut region from the mesh.

    Parameters
    ----------
    mesh : object
        FESOM mesh object
    nlevels : array
        array of size `elem` with number of levels for each element.
        Usually read from *mesh.diag.nc file (`nlevels`) field.
    box : list
        Coordinates of the box in [-180 180 -90 90] format.
        Default set to [13, 30, 53, 66], Baltic Sea.
    depth : float
        depth

    Returns
    -------
    elem_no_nan : array
        elements that belong to the region defined by `box`.
    no_nan_triangles : array
        boolian array of size elem2d with True for elements 
        that belong to the region defines by `box`.
    """

    nlevels = np.array([nlevels] * 3).transpose()
    left, right, down, up = box
    ind_depth = ind_for_depth(depth, mesh)
    elem2 = mesh.elem
    xx = mesh.x2[elem2]
    yy = mesh.y2[elem2]
    dind = ind_for_depth(depth, mesh)

    mask = (nlevels > dind) & (xx >= left) & (xx <= right) & (yy >= down) & (yy <= up)

    mask_elem = mask.mean(axis=1)
    mask_elem[mask_elem != 1] = np.nan

    no_nan_triangles = np.invert(np.isnan(mask_elem))
    elem_no_nan = elem2[no_nan_triangles, :]

    return elem_no_nan, no_nan_triangles

def select_slices(dataset, variable,  mesh, records, depth, continuous=False):

    if depth != None:
        dind = ind_for_depth(depth, mesh)

    if records == -1:
        data = dataset[variable]
    elif isinstance(records, slice):
        if (records.step == None) and (continuous==True):
            data = dataset[variable][records,:]
        elif (records.step != None) and (continuous == False):
            data = dataset[variable].isel(time=records)
        elif (records.step != None) and (continuous==True):
            raise ValueError("You set `continuous` to True, but the step in the slice is not None.")
    elif isinstance(records, list):
        if continuous==True:
            raise ValueError("You set `continuous` to True, lists are not allowed.")
        data = dataset[variable].isel(time=records)
    else:
        raise ValueError("Records should be ether -1 or instance of a list or a slice.")

    if ("nz1" in dataset.dims) and (depth != None):
        if (continuous==True) and len(data.dims == 3) and (dind.step == None):
            data = data[:,:,dind]
        elif (continuous==True) and len(data.dims == 2) and (dind.step == None):
            data = data[:,dind]
        elif (continuous==False):
            data = data.isel(nz1=dind)
    elif ("nz" in dataset.dims) and (depth != None):
        if (continuous==True) and len(data.dims == 3) and (dind.step == None):
            data = data[:,:,dind]
        elif (continuous==True) and len(data.dims == 2) and (dind.step == None):
            data = data[:,dind]
        elif (continuous==False):
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
    continuous = False,
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
    records: int
        number of time steps to be considered for aggregation
        If 1 - only the first record will be taken, if, for example, 5 then
        five time steps will be taken for aggregation.
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
    if records == 0:
        records = 1

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
        print("Model depth: {}".format(abs(mesh.zlev[dind])))
    else:
        print("Depth is None, 3d field will be returned")

    dataset = xr.open_mfdataset(paths, **kwargs)
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
