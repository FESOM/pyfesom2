# -*- coding: utf-8 -*-
#
# This file is part of pyfesom2
# Original code by Dmitry Sidorenko, Nikolay Koldunov, Lukrecia Stulic, 
# Qiang Wang, Sergey Danilov and Patrick Scholz
#

import json
import math as mt
from collections import OrderedDict

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pkg_resources
import shapely
import xarray as xr
from cmocean import cm as cmo

try:
    import cartopy.feature as cfeature
except ImportError:
    print("Cartopy is not installed, plotting is not available.")


def scalar_r2g(al, be, ga, rlon, rlat):
    """
    Converts rotated coordinates to geographical coordinates.

    Parameters
    ----------
    al : float
        alpha Euler angle
    be : float
        beta Euler angle
    ga : float
        gamma Euler angle
    rlon : array
        1d array of longitudes in rotated coordinates
    rlat : array
        1d araay of latitudes in rotated coordinates

    Returns
    -------
    lon : array
        1d array of longitudes in geographical coordinates
    lat : array
        1d array of latitudes in geographical coordinates

    """

    rad = mt.pi / 180
    al = al * rad
    be = be * rad
    ga = ga * rad
    rotate_matrix = np.zeros(shape=(3, 3))
    rotate_matrix[0, 0] = np.cos(ga) * np.cos(al) - np.sin(ga) * np.cos(be) * np.sin(al)
    rotate_matrix[0, 1] = np.cos(ga) * np.sin(al) + np.sin(ga) * np.cos(be) * np.cos(al)
    rotate_matrix[0, 2] = np.sin(ga) * np.sin(be)
    rotate_matrix[1, 0] = -np.sin(ga) * np.cos(al) - np.cos(ga) * np.cos(be) * np.sin(
        al
    )
    rotate_matrix[1, 1] = -np.sin(ga) * np.sin(al) + np.cos(ga) * np.cos(be) * np.cos(
        al
    )
    rotate_matrix[1, 2] = np.cos(ga) * np.sin(be)
    rotate_matrix[2, 0] = np.sin(be) * np.sin(al)
    rotate_matrix[2, 1] = -np.sin(be) * np.cos(al)
    rotate_matrix[2, 2] = np.cos(be)

    rotate_matrix = np.linalg.pinv(rotate_matrix)

    rlat = rlat * rad
    rlon = rlon * rad

    # Rotated Cartesian coordinates:
    xr = np.cos(rlat) * np.cos(rlon)
    yr = np.cos(rlat) * np.sin(rlon)
    zr = np.sin(rlat)

    # Geographical Cartesian coordinates:
    xg = rotate_matrix[0, 0] * xr + rotate_matrix[0, 1] * yr + rotate_matrix[0, 2] * zr
    yg = rotate_matrix[1, 0] * xr + rotate_matrix[1, 1] * yr + rotate_matrix[1, 2] * zr
    zg = (
        rotate_matrix[2, 0] * xr + rotate_matrix[2, 1] * yr + rotate_matrix[2, 2] * zr
    )  
    
    # Geographical coordinates:
    lat = np.arcsin(zg)
    lon = np.arctan2(yg, xg)

    a = np.where((np.abs(xg) + np.abs(yg)) == 0)
    if a:
        lon[a] = 0

    lat = lat / rad
    lon = lon / rad

    return (lon, lat)


def scalar_g2r(al, be, ga, lon, lat):
    """
    Converts geographical coordinates to rotated coordinates.

    Parameters
    ----------
    al : float
        alpha Euler angle
    be : float
        beta Euler angle
    ga : float
        gamma Euler angle
    lon : array
        1d array of longitudes in geographical coordinates
    lat : array
        1d array of latitudes in geographical coordinates

    Returns
    -------
    rlon : array
        1d array of longitudes in rotated coordinates
    rlat : array
        1d araay of latitudes in rotated coordinates
    """

    rad = mt.pi / 180
    al = al * rad
    be = be * rad
    ga = ga * rad

    rotate_matrix = np.zeros(shape=(3, 3))

    rotate_matrix[0, 0] = np.cos(ga) * np.cos(al) - np.sin(ga) * np.cos(be) * np.sin(al)
    rotate_matrix[0, 1] = np.cos(ga) * np.sin(al) + np.sin(ga) * np.cos(be) * np.cos(al)
    rotate_matrix[0, 2] = np.sin(ga) * np.sin(be)
    rotate_matrix[1, 0] = -np.sin(ga) * np.cos(al) - np.cos(ga) * np.cos(be) * np.sin(
        al
    )
    rotate_matrix[1, 1] = -np.sin(ga) * np.sin(al) + np.cos(ga) * np.cos(be) * np.cos(
        al
    )
    rotate_matrix[1, 2] = np.cos(ga) * np.sin(be)
    rotate_matrix[2, 0] = np.sin(be) * np.sin(al)
    rotate_matrix[2, 1] = -np.sin(be) * np.cos(al)
    rotate_matrix[2, 2] = np.cos(be)

    lat = lat * rad
    lon = lon * rad

    # geographical Cartesian coordinates:
    xr = np.cos(lat) * np.cos(lon)
    yr = np.cos(lat) * np.sin(lon)
    zr = np.sin(lat)

    # rotated Cartesian coordinates:
    xg = rotate_matrix[0, 0] * xr + rotate_matrix[0, 1] * yr + rotate_matrix[0, 2] * zr
    yg = rotate_matrix[1, 0] * xr + rotate_matrix[1, 1] * yr + rotate_matrix[1, 2] * zr
    zg = rotate_matrix[2, 0] * xr + rotate_matrix[2, 1] * yr + rotate_matrix[2, 2] * zr

    # rotated coordinates:
    rlat = np.arcsin(zg)
    rlon = np.arctan2(yg, xg)

    a = np.where((np.abs(xg) + np.abs(yg)) == 0)
    if a:
        lon[a] = 0

    rlat = rlat / rad
    rlon = rlon / rad

    return (rlon, rlat)


def vec_rotate_r2g(al, be, ga, lon, lat, urot, vrot, flag):
    """
    Rotate vectors from rotated coordinates to geographical coordinates.

    Parameters
    ----------
    al : float
        alpha Euler angle
    be : float
        beta Euler angle
    ga : float
        gamma Euler angle
    lon : array
        1d array of longitudes in rotated or geographical coordinates (see flag parameter)
    lat : array
        1d array of latitudes in rotated or geographical coordinates (see flag parameter)
    urot : array
        1d array of u component of the vector in rotated coordinates
    vrot : array
        1d array of v component of the vector in rotated coordinates
    flag : 1 or 0
        flag=1  - lon,lat are in geographical coordinate
        flag=0  - lon,lat are in rotated coordinate

    Returns
    -------
    u : array
        1d array of u component of the vector in geographical coordinates
    v : array
        1d array of v component of the vector in geographical coordinates

    """

    #   first get another coordinate
    if flag == 1:
        (rlon, rlat) = scalar_g2r(al, be, ga, lon, lat)
    else:
        rlon = lon
        rlat = lat
        (lon, lat) = scalar_r2g(al, be, ga, rlon, rlat)

    #   then proceed...
    rad = mt.pi / 180
    al = al * rad
    be = be * rad
    ga = ga * rad

    rotate_matrix = np.zeros(shape=(3, 3))
    rotate_matrix[0, 0] = np.cos(ga) * np.cos(al) - np.sin(ga) * np.cos(be) * np.sin(al)
    rotate_matrix[0, 1] = np.cos(ga) * np.sin(al) + np.sin(ga) * np.cos(be) * np.cos(al)
    rotate_matrix[0, 2] = np.sin(ga) * np.sin(be)
    rotate_matrix[1, 0] = -np.sin(ga) * np.cos(al) - np.cos(ga) * np.cos(be) * np.sin(
        al
    )
    rotate_matrix[1, 1] = -np.sin(ga) * np.sin(al) + np.cos(ga) * np.cos(be) * np.cos(
        al
    )
    rotate_matrix[1, 2] = np.cos(ga) * np.sin(be)
    rotate_matrix[2, 0] = np.sin(be) * np.sin(al)
    rotate_matrix[2, 1] = -np.sin(be) * np.cos(al)
    rotate_matrix[2, 2] = np.cos(be)

    rotate_matrix = np.linalg.pinv(rotate_matrix)
    rlat = rlat * rad
    rlon = rlon * rad
    lat = lat * rad
    lon = lon * rad

    #   vector in rotated Cartesian
    txg = -vrot * np.sin(rlat) * np.cos(rlon) - urot * np.sin(rlon)
    tyg = -vrot * np.sin(rlat) * np.sin(rlon) + urot * np.cos(rlon)
    tzg = vrot * np.cos(rlat)

    #   vector in geo Cartesian
    txr = (
        rotate_matrix[0, 0] * txg
        + rotate_matrix[0, 1] * tyg
        + rotate_matrix[0, 2] * tzg
    )
    tyr = (
        rotate_matrix[1, 0] * txg
        + rotate_matrix[1, 1] * tyg
        + rotate_matrix[1, 2] * tzg
    )
    tzr = (
        rotate_matrix[2, 0] * txg
        + rotate_matrix[2, 1] * tyg
        + rotate_matrix[2, 2] * tzg
    )

    #   vector in geo coordinate
    v = (
        -np.sin(lat) * np.cos(lon) * txr
        - np.sin(lat) * np.sin(lon) * tyr
        + np.cos(lat) * tzr
    )
    u = -np.sin(lon) * txr + np.cos(lon) * tyr

    u = np.array(u)
    v = np.array(v)

    return (u, v)


def tunnel_fast1d(latvar, lonvar, lonlat):
    """
    Find closest point in a set of (lat,lon) points to specified pointd.

    Parameters:
    -----------
        latvar : ndarray
            1d array with lats
        lonvar : ndarray
            1d array with lons
        lonlat : ndarray
            2d array with the shape of [2, number_of_point], 
            that contain coordinates of the points

    Returns:
    --------
        node : int
            node number of the closest point

    Taken from here http://www.unidata.ucar.edu/blogs/developer/en/entry/accessing_netcdf_data_by_coordinates
    and modifyed for 1d
    """

    rad_factor = np.pi / 180.0  # for trignometry, need angles in radians
    # Read latitude and longitude from file into numpy arrays
    latvals = latvar[:] * rad_factor
    lonvals = lonvar[:] * rad_factor

    # Compute numpy arrays for all values, no loops
    clat, clon = np.cos(latvals), np.cos(lonvals)
    slat, slon = np.sin(latvals), np.sin(lonvals)

    clat_clon = clat * clon
    clat_slon = clat * slon

    lat0_rad = lonlat[1, :] * rad_factor
    lon0_rad = lonlat[0, :] * rad_factor

    delX_pre = np.cos(lat0_rad) * np.cos(lon0_rad)
    delY_pre = np.cos(lat0_rad) * np.sin(lon0_rad)
    delZ_pre = np.sin(lat0_rad)

    nodes = np.zeros((lonlat.shape[1]))
    for i in range(lonlat.shape[1]):
        delX = delX_pre[i] - clat_clon
        delY = delY_pre[i] - clat_slon
        delZ = delZ_pre[i] - slat
        dist_sq = delX ** 2 + delY ** 2 + delZ ** 2
        minindex_1d = dist_sq.argmin()  # 1D index of minimum element
        node = np.unravel_index(minindex_1d, latvals.shape)
        nodes[i] = node[0]

    return nodes


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.
    Source: https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    Parameters
    ----------
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)

    return newcmap


def mask_ne(lonreg2, latreg2):
    """ Mask earth from lon/lat data using Natural Earth.

    Parameters
    ----------
    lonreg2: float, np.array
        2D array of longitudes
    latreg2: float, np.array
        2D array of latitudes

    Returns
    -------
    m2: bool, np.array
        2D mask with True where the ocean is.
    """
    nearth = cfeature.NaturalEarthFeature("physical", "ocean", "50m")
    main_geom = [contour for contour in nearth.geometries()][0]

    mask = shapely.vectorized.contains(main_geom, lonreg2, latreg2)
    m2 = np.where(((lonreg2 == -180.0) & (latreg2 > 71.5)), True, mask)
    m2 = np.where(
        ((lonreg2 == -180.0) & (latreg2 < 70.95) & (latreg2 > 68.96)), True, m2
    )
    m2 = np.where(((lonreg2 == 180.0) & (latreg2 > 71.5)), True, mask)
    m2 = np.where(
        ((lonreg2 == 180.0) & (latreg2 < 70.95) & (latreg2 > 68.96)), True, m2
    )
    # m2 = np.where(
    #        ((lonreg2 == 180.0) & (latreg2 > -75.0) & (latreg2 < 0)), True, m2
    #    )
    m2 = np.where(((lonreg2 == -180.0) & (latreg2 < 65.33)), True, m2)
    m2 = np.where(((lonreg2 == 180.0) & (latreg2 < 65.33)), True, m2)

    return ~m2


def mask_from_file(filename, name, mesh):
    """Create mask from geojson file

    Parameters:
    ----------=
        filename: str
            path to the geojson file.
            File should have several features
        name: str
            name of the feature, that will be our mask
        mesh: mesh object
            pyfesom2 mesh object

    Returns:
    --------
        mask: numpy array
            boolean vector of the shape of 2D data.

    """
    with open(filename) as f:
        features = json.load(f)["features"]
    for feature in features:
        if feature["properties"]["name"] == name:
            geom = shapely.geometry.shape(feature["geometry"])
            mask = shapely.vectorized.contains(geom, mesh.x2, mesh.y2)
    return mask


def get_mask(mesh, region):
    """Return mask.

    Parameters:
    -----------
    mesh: mesh object
        pyfesom2 mesh object
    name: str
        name of the region.
        Available regions:
            Ocean Basins:
                "Atlantic_Basin"
                "Pacific_Basin"
                "Indian_Basin"
                "Arctic_Basin"
                "Southern_Ocean_Basin"
                "Mediterranean_Basin"
                "Global Ocean"
                "Global Ocean 65N to 65S"
                "Global Ocean 15S to 15N"
            MOC Basins:
                "Atlantic_MOC"
                "IndoPacific_MOC"
                "Pacific_MOC"
                "Indian_MOC"
            Nino Regions:
                "Nino 3.4"
                "Nino 3"
                "Nino 4"
            Arctic Ocean regions:
                "Amerasian basin"
                "Eurasian basin"
    Returns:
    --------
        mask: numpy array
            boolean or integer (for AO regions) vector of the shape of 2D data.
    """

    MOCBasins = ["Atlantic_MOC", "IndoPacific_MOC", "Pacific_MOC", "Indian_MOC"]
    NinoRegions = ["Nino 3.4", "Nino 3", "Nino 4"]
    oceanBasins = [
        "Atlantic_Basin",
        "Pacific_Basin",
        "Indian_Basin",
        "Arctic_Basin",
        "Southern_Ocean_Basin",
        "Mediterranean_Basin",
        "Global Ocean",
        "Global Ocean 65N to 65S",
        "Global Ocean 15S to 15N",
    ]

    if region == "Amerasian basin":
        ind_AB1 = np.where((mesh.x2 >= -100) & (mesh.x2 < -80) & (mesh.y2 >= 80))
        ind_AB2 = np.where((mesh.x2 >= -180) & (mesh.x2 < -100) & (mesh.y2 > 66))
        ind_AB3 = np.where((mesh.x2 < 180) & (mesh.x2 > 140) & (mesh.y2 > 66))

        mask = np.hstack((ind_AB1[0], ind_AB2[0], ind_AB3[0]))

    elif region == "Eurasian basin":
        ind_EB1 = np.where((mesh.x2 > -80) & (mesh.x2 < 100) & (mesh.y2 > 80))
        ind_EB2 = np.where((mesh.x2 > 100) & (mesh.x2 < 140) & (mesh.y2 > 66))

        mask = np.hstack((ind_EB1[0], ind_EB2[0]))

    elif region in MOCBasins:
        filename = pkg_resources.resource_filename(
            __name__, "geojson/MOCBasins.geojson"
        )
        mask = mask_from_file(filename, region, mesh)

    elif region in NinoRegions:
        filename = pkg_resources.resource_filename(
            __name__, "geojson/NinoRegions.geojson"
        )
        mask = mask_from_file(filename, region, mesh)

    elif region in oceanBasins:
        filename = pkg_resources.resource_filename(
            __name__, "geojson/oceanBasins.geojson"
        )
        mask = mask_from_file(filename, region, mesh)

    else:
        raise ValueError(f"Name {region} is not in the get_mask function.")

    return mask


def compute_face_coords(mesh):
    """ Compute coordinates of elements (triangles)

    Parameters:
    -----------
    mesh: mesh object
        fesom mesh object

    Returns:
    --------
    face_x: numpy array
        x coordinates
    face_y: numpy array
        y coordinates
    """
    first_mean = mesh.x2[mesh.elem].mean(axis=1)
    j = np.where(np.abs(mesh.x2[mesh.elem][:, 0] - first_mean) > 100)[0]
    cyclic_elems = mesh.x2[mesh.elem][j].copy()
    new_means = np.where(cyclic_elems > 0, cyclic_elems, cyclic_elems + 360).mean(
        axis=1
    )
    new_means[new_means > 180] = new_means[new_means > 180] - 360
    first_mean[j] = new_means
    face_x = first_mean
    face_y = mesh.y2[mesh.elem].mean(axis=1)
    return face_x, face_y


def set_standard_attrs(da):
    """ Add standard attributed to xarray DataArray"""
    da.coords["lat"].attrs = OrderedDict(
        [
            ("standard_name", "latitude"),
            ("units", "degrees_north"),
            ("axis", "Y"),
            ("long_name", "latitude"),
            ("out_name", "lat"),
            ("stored_direction", "increasing"),
            ("type", "double"),
            ("valid_max", "90.0"),
            ("valid_min", "-90.0"),
        ]
    )
    da.coords["lon"].attrs = OrderedDict(
        [
            ("standard_name", "longitude"),
            ("units", "degrees_east"),
            ("axis", "X"),
            ("long_name", "longitude"),
            ("out_name", "lon"),
            ("stored_direction", "increasing"),
            ("type", "double"),
            ("valid_max", "180.0"),
            ("valid_min", "-180.0"),
        ]
    )
    da.coords["depth_coord"].attrs = OrderedDict(
        [
            ("standard_name", "depth"),
            ("units", "m"),
            ("axis", "Z"),
            ("long_name", "ocean depth coordinate"),
            ("out_name", "lev"),
            ("positive", "down"),
            ("stored_direction", "increasing"),
            ("valid_max", "12000.0"),
            ("valid_min", "0.0"),
        ]
    )
    da.coords["time"].attrs = OrderedDict(
        [
            ("standard_name", "time"),
            ("axis", "T"),
            ("long_name", "time"),
            ("out_name", "time"),
            ("stored_direction", "increasing"),
        ]
    )
    da.coords["time"].encoding["units"] = "days since '1900-01-01'"

    return da


def cut_region(mesh, box):
    """
    Cut region from the mesh.

    Parameters
    ----------
    mesh : object
        FESOM mesh object
    box : list
        Coordinates of the box in [-180 180 -90 90] format.
        Default set to [13, 30, 53, 66], Baltic Sea.

    Returns
    -------
    elem_no_nan : array
        elements that belong to the region defined by `box`.
    """
    x_on_triangles = mesh.x2[mesh.elem]
    y_on_triangles = mesh.y2[mesh.elem]

    left, right, down, up = box

    selection = (
        (x_on_triangles >= left)
        & (x_on_triangles <= right)
        & (y_on_triangles >= down)
        & (y_on_triangles <= up)
    )
    #     indd = np.where((mesh.x2>=left) & (mesh.x2<=right) & (mesh.y2>=down) & (mesh.y2<=up))
    # vvv = ((dd>=0) & (xx>=7.5) & (xx<=8.2) & (yy>=54) & (yy<=54.5))
    mask = selection.mean(axis=1)

    mask[mask != 1] = np.nan

    no_nan_triangles = np.invert(np.isnan(mask))

    elem_no_nan = mesh.elem[no_nan_triangles, :]

    return elem_no_nan


def get_cmap(cmap=None):
    """Get the color map.

    Parameters
    ----------
    cmap: str, mpl.colors.Colormap
        The colormap can be provided as the name (should be in matplotlib or cmocean colormaps),
        or as matplotlib colormap object.

    Returns
    -------
    colormap:mpl.colors.Colormap
        Matplotlib colormap object.

    """
    if cmap:
        if isinstance(cmap, (mpl.colors.Colormap)):
            colormap = cmap
        elif cmap in cmo.cmapnames:
            colormap = cmo.cmap_d[cmap]
        elif cmap in plt.colormaps():
            colormap = plt.get_cmap(cmap)
        else:
            raise ValueError(
                "Get unrecognised name for the colormap `{}`. Colormaps should be from standard matplotlib set of from cmocean package.".format(
                    cmap
                )
            )
    else:
        colormap = plt.get_cmap("Spectral_r")

    return colormap


def get_no_cyclic(mesh, elem_no_nan):
    """Compute non cyclic elements of the mesh."""
    d = mesh.x2[elem_no_nan].max(axis=1) - mesh.x2[elem_no_nan].min(axis=1)
    no_cyclic_elem = [i for (i, val) in enumerate(d) if val < 100]

    return no_cyclic_elem
