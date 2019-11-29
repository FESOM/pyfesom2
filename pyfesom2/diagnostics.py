# -*- coding: utf-8 -*-
#
# This file is part of pyfesom2
# Original code by Nikolay Koldunov, Dmitry Sidorenko and Qiang Wang (2019)

import numpy as np
import xarray as xr


def add_timedim(data, date="1970-01-01"):
    """Add a dummy time dimension."""
    if isinstance(data, xr.DataArray):
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

