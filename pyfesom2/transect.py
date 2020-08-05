# -*- coding: utf-8 -*-
#
# This file is part of pyfesom2
# Original code by Dmitry Sidorenko, Nikolay Koldunov,
# Qiang Wang, Sergey Danilov and Patrick Scholz
#


import math

import numpy as np
import pyproj

from .load_mesh_data import ind_for_depth
from .ut import tunnel_fast1d, vec_rotate_r2g

g = pyproj.Geod(ellps="WGS84")


def transect_get_lonlat(lon_start, lat_start, lon_end, lat_end, npoints=30):
    lonlat = g.npts(lon_start, lat_start, lon_end, lat_end, npoints)
    lonlat = np.array(lonlat)
    return lonlat.T


def transect_get_nodes(lonlat, mesh):
    nodes = tunnel_fast1d(mesh.y2, mesh.x2, lonlat)
    return nodes.astype('int')


def transect_get_distance(lonlat):
    (az12, az21, dist) = g.inv(
        lonlat[0, :][0:-1], lonlat[1, :][0:-1], lonlat[0, :][1:], lonlat[1, :][1:]
    )
    dist = dist.cumsum() / 1000
    dist = np.insert(dist, 0, 0)
    return dist


# def transect_get_profile(nodes, mesh):
#     profile = (mesh.n32-1)[nodes,:]
#     return profile


def transect_get_mask(nodes, mesh, lonlat, max_distance):
    (az12, az21, point_dist) = g.inv(
        lonlat[0, :], lonlat[1, :], mesh.x2[nodes], mesh.y2[nodes]
    )
    mask = ~(point_dist < max_distance)
    mask2d = np.repeat(mask, mesh.zlev.shape[0] - 1).reshape(
        (len(nodes), mesh.zlev.shape[0] - 1)
    )
    return mask2d


def transect_get_data(data3d, nodes, mask2d=None):
    nlevels = data3d.shape[1]
    transect_data = data3d[nodes, :]
    transect_data = np.ma.masked_where(transect_data == 0, transect_data)
    if type(mask2d) is np.ndarray:
        mask_nlev = mask2d.shape[1]
        if nlevels > mask_nlev:
            mask2d = np.hstack((mask2d, mask2d[:, -1:]))
        transect_data = np.ma.masked_where(mask2d, transect_data)
    return transect_data

def get_transect(data, mesh, lonlat, max_distance=1e6):
    """Create transect from 3D data and collection of points.
    
    Parameters
    ----------
    data: np array, xarray
        3D fesom data
    mesh: mesh object
        FESOM2 mesh object
    lonlat: numpy array
        Array of shape (2, npoints), that consist lons and lats.
        can be obtained by the `transect_get_lonlat` function.
    max_distance:
        maximum distance to still consider the points for using in the transect.

    Returns
    -------
    dist: numpy array
        distances of each point from the first coordinates.
    transect_data: numpy array
        2D array of shape (npoints, nlevels)
    """
    nodes = transect_get_nodes(lonlat, mesh)
    dist = transect_get_distance(lonlat)
    mask2d = transect_get_mask(nodes, mesh, lonlat, max_distance)
    transect_data = transect_get_data(data, nodes, mask2d)
    return dist, transect_data

def transect_uv(*args, **kwargs):
    raise DeprecationWarning("The transect_uv function is deprecated. Use get_transect_uv instead.")

def get_transect_uv(
    udata3d,
    vdata3d,
    mesh,
    lonlat,
    abg=[50, 15, -90],
    max_distance=None,
    myangle=0,
):
    """Get transect for UV data.

    Parameters
    
    """
    # lonlat = transect_get_lonlat(
    #     lon_start, lat_start, lon_end, lat_end, npoints=npoints
    # )
    nodes = transect_get_nodes(lonlat, mesh)
    dist = transect_get_distance(lonlat)
    if max_distance:
        mask2d = transect_get_mask(nodes, mesh, lonlat, max_distance)
    else:
        mask2d = None

    u = udata3d[nodes, :]
    v = vdata3d[nodes, :]

    rot_u = []
    rot_v = []
    for i in range(u.shape[1]):
        uu, vv = vec_rotate_r2g(
            abg[0],
            abg[1],
            abg[2],
            mesh.x2[nodes],
            mesh.y2[nodes],
            u[:, i],
            v[:, i],
            flag=1,
        )
        rot_u.append(uu)
        rot_v.append(vv)
    rot_u = np.array(rot_u)
    rot_v = np.array(rot_v)

    if myangle != 0:
        direct = np.rad2deg(np.arctan2(rot_v, rot_u))
        speed_rot = np.hypot(rot_u, rot_v)

        myangle = myangle
        U = speed_rot * np.cos(np.deg2rad(myangle - direct))
        V = speed_rot * np.sin(np.deg2rad(myangle - direct))

        U = np.ma.masked_where(u.T == 0, U)
        V = np.ma.masked_where(v.T == 0, V)
        if type(mask2d) is np.ndarray:
            U = np.ma.masked_where(mask2d.T, U)
            V = np.ma.masked_where(mask2d.T, V)
        rot_u = U
        rot_v = V
    else:
        rot_u = np.ma.masked_where(u.T == 0, rot_u)
        rot_v = np.ma.masked_where(u.T == 0, rot_v)
        if type(mask2d) is np.ndarray:
            rot_u = np.ma.masked_where(mask2d.T, rot_u)
            rot_v = np.ma.masked_where(mask2d.T, rot_v)
        return dist, rot_u.T, rot_v.T

    return dist, rot_u.T, rot_v.T


def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(long).cos(lat2),
                  cos(lat1).sin(lat2) sin(lat1).cos(lat2).cos(long))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float

      Source: https://gist.github.com/jeromer/2005586
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (
        math.sin(lat1) * math.cos(lat2) * math.cos(diffLong)
    )

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing
