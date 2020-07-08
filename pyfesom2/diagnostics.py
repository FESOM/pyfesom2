# -*- coding: utf-8 -*-
#
# This file is part of pyfesom2
# Original code by Dmitry Sidorenko, Nikolay Koldunov, 
# Qiang Wang, Sergey Danilov and Patrick Scholz
#

import os

import numpy as np
import xarray as xr
from pandas.plotting import register_matplotlib_converters

from .load_mesh_data import ind_for_depth
from .ut import compute_face_coords, get_mask

register_matplotlib_converters()


def add_timedim(data, date="1970-01-01"):
    """Add a dummy time dimension."""
    if isinstance(data, xr.DataArray):
        if "time" in data.dims:
            raise ValueError(
                "You trying to add time dimension to the DataArray that already have it. \
The reason migh be that you trying to use 2d variable (e.g. `a_ice`) \
in a function that accepts only 3d variables (e.g. `hovm_data`)"
            )
        timestamp = [np.array(np.datetime64(date, "ns"))]
        data = data.expand_dims({"time": timestamp}, axis=0)
        return data
    else:
        data = np.expand_dims(data, axis=0)
        return data


def ice_ext(data, mesh, hemisphere="N", threshhold=0.15, attrs={}):
    """ Compute sea ice extent.

    Parameters
    ----------
    data: xarray.DataArray, numpy.array
        Input data, that can be ether xarray data, or just numpy array,
        with values of `a_ice`.
        Input should be 2d (time, nodes)
    mesh: mesh object
        FESOM2 mesh object.
    hemisphere: str
        can be ether 'N' or 'S'
    threshhold: float
        default is 0.15
    attrs: dict
        dictionary of attributes that will be put in the resulting
        xarray DataArray if input was also a xarray DataArray.

    Returns
    -------
    ext: xarray.DataArray, numpy.array
        depends on the input xarray.DataArray or numpy.array will be returned.

    """
    if len(data.shape) == 1:
        data = add_timedim(data)

    if hemisphere == "N":
        varname = "siextentn"
        hemis_mask = mesh.y2 > 0
    else:
        varname = "siextents"
        hemis_mask = mesh.y2 < 0

    if isinstance(data, xr.DataArray):
        data = data.where(data < threshhold, 1)
        data = data.where(data > threshhold)

        ext = (data[:, hemis_mask] * mesh.lump2[hemis_mask]).sum(axis=1)
        da = xr.DataArray(
            ext, dims=["time"], coords={"time": data.time}, name=varname, attrs=attrs
        )
        return da

    else:
        print(data)
        i, j = np.where(data < 0.15)
        data[:] = 1
        data[i, j] = 0
        ext = (data[:, hemis_mask] * mesh.lump2[hemis_mask]).sum(axis=1)
        return ext


def ice_vol(data, mesh, hemisphere="N", attrs={}):
    """ Compute sea ice volume.

    Parameters
    ----------
    data: xarray.DataArray, numpy.array
        Input data, that can be ether xarray data, or just numpy array,
        with values of `m_ice`.
        Input shouls be 2d (time, nodes)
    mesh: mesh object
        FESOM2 mesh object.
    hemisphere: str
        can be ether 'N' or 'S'
    attrs: dict
        dictionary of attributes that will be put in the resulting
        xarray DataArray if input was also a xarray DataArray.

    Returns
    -------
    ext: xarray.DataArray, numpy.array
        depends on the input xarray.DataArray or numpy.array will be returned.

    """
    if len(data.shape) == 1:
        data = add_timedim(data)

    if hemisphere == "N":
        varname = "sivoln"
        hemis_mask = mesh.y2 > 0
    else:
        varname = "sivols"
        hemis_mask = mesh.y2 < 0

    if isinstance(data, xr.DataArray):
        vol = (data[:, hemis_mask] * mesh.lump2[hemis_mask]).sum(axis=1)
        da = xr.DataArray(
            vol, dims=["time"], coords={"time": data.time}, name=varname, attrs=attrs
        )
        return da
    else:
        vol = (data[:, hemis_mask] * mesh.lump2[hemis_mask]).sum(axis=1)
        return vol


def ice_area(data, mesh, hemisphere="N", attrs={}):
    """ Compute sea ice volume.

    Parameters
    ----------
    data: xarray.DataArray, numpy.array
        Input data, that can be ether xarray data, or just numpy array,
        with values of `a_ice`.
        Input shouls be 2d (time, nodes)
    mesh: mesh object
        FESOM2 mesh object.
    hemisphere: str
        can be ether 'N' or 'S'
    attrs: dict
        dictionary of attributes that will be put in the resulting
        xarray DataArray if input was also a xarray DataArray.

    Returns
    -------
    ext: xarray.DataArray, numpy.array
        depends on the input xarray.DataArray or numpy.array will be returned.

    """

    if len(data.shape) == 1:
        data = add_timedim(data)

    if hemisphere == "N":
        varname = "siarean"
        hemis_mask = mesh.y2 > 0
    else:
        varname = "siareas"
        hemis_mask = mesh.y2 < 0

    if isinstance(data, xr.DataArray):
        vol = (data[:, hemis_mask] * mesh.lump2[hemis_mask]).sum(axis=1)
        da = xr.DataArray(
            vol, dims=["time"], coords={"time": data.time}, name=varname, attrs=attrs
        )
        return da
    else:
        area = (data[:, hemis_mask] * mesh.lump2[hemis_mask]).sum(axis=1)
        return area


def get_meshdiag(mesh, meshdiag=None, runid="fesom"):
    """Get mesh diagnostic file.

    If the name is not provided, search in the mesh direcrory.
    """
    if meshdiag:
        if os.path.exists(meshdiag):
            diag = xr.open_dataset(meshdiag)
        else:
            raise Exception("File {} does not exist.".format(meshdiag))
    elif not meshdiag:
        path_to_meshdiag = os.path.join(mesh.path, "{}.mesh.diag.nc".format(runid))
        if os.path.exists(path_to_meshdiag):
            diag = xr.open_dataset(path_to_meshdiag)
        else:
            raise Exception(
                "File {} does not exist. Please provide explicit path\
                to diag file (e.g. fesom.mesh.diag), or put it to the mesh directory.\
                            ".format(
                    path_to_meshdiag
                )
            )
    return diag


def hovm_data(data, mesh, meshdiag=None, runid="fesom", mask=None):
    """Calculate data for hovmoller diagram.

    Use 3d tracer variable (on nodes) to calculate weighted
     mean value for each vertical layer for every time step.

     Parameters
     ----------
     data: xarray.DataArray, numpy.array
        Input data, that can be ether xarray data, or just numpy array,
        with values of 3d tracer variable (on nodes).
        Input should be 3d (time, nodes, levels(nz1))
        The 2D input (to calculate value for only one time step)
        in form of (nodes, levels(nz1)) is possible, but not recomended :)
    mesh: mesh object
        FESOM2 mesh object.
    meshdiag: str
        path to *mesh.diag.nc file, that is created during fesom cold start.
    runid: str
        name of the run. Usually just `fesom`.
    mask: array of bool
        array of boolian values of the same shape as 2D variable.
        True where data are selected.

    Returns
    -------
    time series: xarray.DataArray
        time series of area weighted vertical profiles of scalar values.
    """

    if len(data.shape) == 2:
        data = add_timedim(data)

    diag = get_meshdiag(mesh, meshdiag, runid)
    nod_area = diag.rename_dims({"nl": "nz1", "nod_n": "nod2"}).nod_area
    nod_area.load()
    if mask is not None:
        nod_area = nod_area[:, mask]
        data = data[:, mask, :]

    if isinstance(data, xr.DataArray):
        nod_area = nod_area.where(nod_area != 0)
        hdg_total = (data * nod_area[:-1, :].T).sum(dim="nod2")
        hdg_variable = hdg_total / (nod_area[:-1, :].T).sum(axis=0)
        hdg_variable = hdg_variable.compute()
    else:
        hdg_total = (data * nod_area[:-1, :].T.data).sum(axis=1)
        hdg_variable = hdg_total / (nod_area[:-1, :].T).sum(axis=0).data

    return hdg_variable


def select_depths(uplow, mesh):
    """ Select indexes of depths between upper and lower bound.

    Parameters
    ----------
    uplow: list
        if None, all depths will be selected
        if e.g. [2000, 'depth'] all from model depth closest to 200 down to depth will be selected
        if e.g. [0, 700] all between 0 and closest depth to 700 will be selected.
        if e.g. [500, 500] only model level closest to 500 will be selected.
    mesh: mesh object
        FESOM2 mesh object.

    Returns
    -------
    indexes: range
        range of depth indexes.
    """
    if not uplow:
        indexes = range(mesh.nlev - 1)
        return indexes
    elif uplow[1] == "bottom":
        upper = ind_for_depth(uplow[0], mesh)
        lower = mesh.nlev - 1
        print(f"Upper depth: {mesh.zlev[upper]}, Lower depth: bottom")
        indexes = range(upper, lower)
        return indexes
    else:
        upper = ind_for_depth(uplow[0], mesh)
        lower = ind_for_depth(uplow[1], mesh)
        if lower >= mesh.nlev - 2:
            lower = mesh.nlev - 2
        print(f"Upper depth: {mesh.zlev[upper]}, Lower depth: {mesh.zlev[lower]}")
        indexes = range(upper, lower + 1)
        return indexes


def volmean_data(data, mesh, uplow=None, meshdiag=None, runid="fesom", mask=None):
    """Calculate volume weighted mean over the range of depths.

    Parameters
     ----------
     data: xarray.DataArray, numpy.array
        Input data, that can be ether xarray data, or just numpy array,
        with values of 3d tracer variable (on nodes).
        Input should be 3d (time, nodes, levels(nz1))
        The 2D input (to calculate value for only one time step)
        in form of (nodes, levels(nz1)) is possible, but not recomended :)
    mesh: mesh object
        FESOM2 mesh object.
    uplow: list
        if None, all depths will be selected
        if e.g. [2000, 'depth'] all from model depth closest to 200 down to depth will be selected
        if e.g. [0, 700] all between 0 and closest depth to 700 will be selected.
        if e.g. [500, 500] only model level closest to 500 will be selected.
    meshdiag: str
        path to *mesh.diag.nc file, that is created during fesom cold start.
    runid: str
        name of the run. Usually just `fesom`.
    mask: array of bool
        array of boolian values of the same shape as 2D variable.
        True where data are selected.

    Returns
    -------
    time series: xarray.DataArray
        time series (or one point) of volume weighted scalar.
    """
    if len(data.shape) == 2:
        data = add_timedim(data)

    diag = get_meshdiag(mesh, meshdiag, runid)
    nod_area = diag.rename_dims({"nl": "nz1", "nod_n": "nod2"}).nod_area
    nod_area.load()
    #     nod_area = nod_area.where(nod_area != 0)
    delta_z = np.abs(np.diff(mesh.zlev))

    indexes = select_depths(uplow, mesh)

    total_t = 0.0
    total_v = 0.0
    # we calculate layer by layer
    if mask is not None:
        nod_area = nod_area[:, mask]
        data = data[:, mask, :]

    for i in indexes:
        nod_area_at_level = np.ma.masked_equal(nod_area[i, :].data, 0)
        aux = (data[:, :, i] * nod_area_at_level[:]).sum(axis=1)
        if not np.ma.is_masked(nod_area_at_level[:].sum()):
            total_t = total_t + aux * delta_z[i]
            total_v = total_v + nod_area_at_level[:].sum() * delta_z[i]

    return total_t / total_v


def xmoc_data(
    mesh,
    data,
    nlats=91,
    mask="Global Ocean",
    return_masked=True,
    meshdiag=None,
    el_area=None,
    nlevels=None,
    face_x=None,
    face_y=None,
):
    """ Compute moc for selected region.

    Parameters:
    -----------
    mesh: mesh object
        pyfesom mesh object
    data: numpy array, xarray DataArray
        3D field of w(nod2d, levels)
    nlats: int
        number of latitude bins
    mask: str, numpy array
        can be ether name of the mask from the list available in "pyfesom.get_mask"
        function, or the mask itself as 1d array of the nod2d size.
    return_masked: bool
        control if we return masked or pure moc array.
    meshdiag: str
        path to the mesh diag file. If None try to find it in mesh folder.
    el_area: numpy array
        area of elements, size elem2d. If None, will be computed.
    nlevels: numpu array
        number of levels for each element, size elem2d. If None, will be extracted from mesh diag file.
    face_x: numpy array
        x coordinates of centers of elements, size elem2d. If None, will be computed.
    face_y: numpy array
        y coordinates of centers of elements, size elem2d. If None, will be computed.

    Returns:
    --------
    lats: numpy array
        latitude bins
    moc_masked: numpy array
       masked array with moc values.

    """

    if len(data.shape) == 3:
        raise ValueError(
            "You have 3 dimensions in input data, xmoc_data \
                          accepts only one 3D field of w(nodes, levels)."
        )

    # option to provide el_area and nlevelsm not to load them every time
    if (el_area is None) or (nlevels is None):
        meshdiag = get_meshdiag(mesh, meshdiag=meshdiag)
        el_area = meshdiag["elem_area"][:]
        nlevels = meshdiag["nlevels"][:] - 1

    # option to provide precomputed face_x and face_y
    if (face_x is None) or (face_y is None):
        face_x, face_y = compute_face_coords(mesh)

    # can provide mask or mask name
    if isinstance(mask, str):
        mask = get_mask(mesh, mask)
    else:
        mask = mask

    nlats = nlats
    lats = np.linspace(-90, 90, nlats)
    dlat = lats[1] - lats[0]
    # allocate moc array
    moc = np.zeros([mesh.nlev, nlats])
    pos = ((face_y - lats[0]) / dlat).astype("int")

    if isinstance(data, xr.DataArray):
        w = data[:, :].values * mask[:, None]
    else:
        w = data[:, :] * mask[:, None]

    elem_mean = np.sum(w[mesh.elem.T, :], axis=0) / 3.0 * 1.0e-6
    elem_mean_weigh = elem_mean * el_area.values[:, None]
    for i in range(0, mesh.nlev):
        not_calc = np.where(i >= nlevels)[0]
        elem_mean_weigh[not_calc, i] = np.nan

    for k in range(pos.min(), pos.max() + 1):
        moc[:, k] = np.nansum(elem_mean_weigh[pos[:] == k, :], axis=0)

    i, j = np.where(moc.T == 0)
    moc_cumsum = np.ma.cumsum(moc[:, ::-1], axis=1)
    moc_proper_order = moc_cumsum[:, ::-1].T * -1
    if return_masked:
        moc_proper_order[i, j] = 0
        moc_masked = np.ma.masked_equal(moc_proper_order, 0)
        moc_final = moc_masked
    else:
        moc_final = moc_proper_order

    return lats, moc_final
