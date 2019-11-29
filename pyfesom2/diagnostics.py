# -*- coding: utf-8 -*-
#
# This file is part of pyfesom2
# Original code by Nikolay Koldunov, Dmitry Sidorenko and Qiang Wang (2019)

import numpy as np
import xarray as xr
import os


def add_timedim(data, date="1970-01-01"):
    """Add a dummy time dimension."""
    if isinstance(data, xr.DataArray):
        if "time" in data.dims:
            raise ValueError(
                "You trying to add time dimension to the DataArray that already have it. \
The reason migh be that you trying to use 2d variable (e.g. `a_ice`) \
in a function that accepts only 3d variables (e.g. `volume_hovm`)"
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
            return diag
        else:
            raise Exception("File {} does not exist.".format(meshdiag))
    elif not meshdiag:
        path_to_meshdiag = os.path.join(mesh.path, "{}.mesh.diag.nc".format(runid))
        if os.path.exists(path_to_meshdiag):
            diag = xr.open_dataset(path_to_meshdiag)
            return diag
        else:
            raise Exception(
                "File {} does not exist. Please provide explicit path\
                             to diag file (e.g. fesom.mesh.diag), or put it to the mesh directory.\
                            ".format(
                    path_to_meshdiag
                )
            )


def hovm_data(data, mesh, meshdiag=None, runid="fesom"):
    """Calculate data for hovmoller diagram.

    Use 3d tracer variable (on nodes) to calculate weighted
     mean value for each vertical layer for every time step.

     Parameters
     ----------
     data: xarray.DataArray, numpy.array
        Input data, that can be ether xarray data, or just numpy array,
        with values of 3d tracer variable (on nodes).
        Input shouls be 3d (time, nodes, levels(nz1))
        The 2D input in form of (nodes, levels(nz1)) is possible, but not recomended :)
    mesh: mesh object
        FESOM2 mesh object.
    meshdiag: str
        path to *mesh.diag.nc file, that is created during fesom cold start.
    runid: str
        name of the run. Usually just `fesom`.
    """

    if len(data.shape) == 2:
        data = add_timedim(data)

    diag = get_meshdiag(mesh, meshdiag)
    nod_area = diag.rename_dims({"nl": "nz1", "nod_n": "nod2"}).nod_area
    if isinstance(data, xr.DataArray):
        hdg_total = (data * nod_area[:-1, :].T).sum(dim="nod2")
        hdg_variable = hdg_total / (nod_area[:-1, :].T).sum(axis=0)
        hdg_variable = hdg_variable.compute()
    else:
        hdg_total = (data * nod_area[:-1, :].T.data).sum(axis=1)
        hdg_variable = hdg_total / (nod_area[:-1, :].T).sum(axis=0).data

    return hdg_variable


def select_depths(uplow, mesh):
    if not uplow:
        indexes = range(mesh.nlev - 1)
        return indexes
    elif uplow[1] == "bottom":
        upper = pf.ind_for_depth(uplow[0], mesh)
        lower = mesh.nlev - 1
        print(f"Upper depth: {mesh.zlev[upper]}, Lower depth: bottom")
        indexes = range(upper, lower)
        return indexes
    else:
        upper = pf.ind_for_depth(uplow[0], mesh)
        lower = pf.ind_for_depth(uplow[1], mesh)
        if lower >= mesh.nlev - 2:
            lower = mesh.nlev - 2
        print(f"Upper depth: {mesh.zlev[upper]}, Lower depth: {mesh.zlev[lower]}")
        indexes = range(upper, lower + 1)
        return indexes


def volume_int(data, mesh, meshdiag=None, runid="fesom", uplow=None):
    if len(data.shape) == 2:
        data = pf.add_timedim(data)

    diag = pf.get_meshdiag(mesh, meshdiag)
    nod_area = diag.rename_dims({"nl": "nz1", "nod_n": "nod2"}).nod_area

    indexes = select_depths(uplow, mesh)

    totalT = 0.0
    totalV = 0.0
    for i in indexes:
        aux = (data[:, :, i] * nod_area[i, :].data).sum(axis=1)
        totalT = totalT + aux * dz[i]
        totalV = totalV + nod_area[i, :].data.sum() * dz[i]

    return totalT / totalV

