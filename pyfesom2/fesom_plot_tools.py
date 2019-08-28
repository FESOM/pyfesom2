# -*- coding: utf-8 -*-
#
# This file is part of pyfesom2
# Original code by Dmitry Sidorenko, 2013
#

import numpy as np
import matplotlib.pyplot as plt

try:
    from mpl_toolkits.basemap import Basemap
except KeyError:
    # dirty hack to avoid KeyError: 'PROJ_LIB' problem with basemap
    import conda
    import os

    conda_file_dir = conda.__file__
    conda_dir = conda_file_dir.split("lib")[0]
    proj_lib = os.path.join(os.path.join(conda_dir, "share"), "proj")
    os.environ["PROJ_LIB"] = proj_lib

    from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
from .regriding import fesom2regular
from netCDF4 import Dataset, MFDataset, num2date
import matplotlib as mpl

# mpl.use('Qt5Agg')
# %matplotlib inline
import matplotlib.pylab as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cmocean import cm as cmo
from matplotlib import cm
import sys, os

# sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
# import pyfesom2 as pf
from cartopy.util import add_cyclic_point

# from scipy.interpolate import griddata
# import scipy.spatial.qhull as qhull
# from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
from cartopy.util import add_cyclic_point

# import xarray as xr
import shapely.vectorized
import joblib
from .transect import *
import matplotlib
from .ut import mask_ne


def ftriplot(
    mesh,
    data2,
    contours,
    cmap=[],
    oce="global",
    do_cbar=True,
    mlabels=[0, 0, 0, 0],
    plabels=[0, 0, 0, 0],
    extend="both",
    data_on_elem=0,
):
    if cmap == []:
        cmap = plt.cm.jet
    if oce == "global":
        data2 = np.copy(data2)

        elem2 = mesh.elem[mesh.no_cyclic_elem, :]

        if data_on_elem == 0:
            d = data2[elem2].mean(axis=1)
        else:
            data2 = data2[mesh.no_cyclic_elem]
            d = data2

        k = [i for (i, val) in enumerate(d) if not np.isnan(val)]
        elem2 = elem2[k, :]

        if data_on_elem == 1:
            data2 = data2[k]

        print("ftriplot, number of dummy points:", len(d) - len(k))
        map = Basemap(projection="robin", lon_0=0)
        x, y = map(mesh.x2, mesh.y2)
        map.drawmapboundary(fill_color="0.9")
        map.drawcoastlines()
        map.drawparallels(np.arange(-90, 90, 30), labels=plabels)  # [1,0,0,0]
        map.drawmeridians(
            np.arange(map.lonmin, map.lonmax + 30, 60), labels=mlabels
        )  # [0,0,0,1]
        # data2[data2>900]=np.nan
        eps = (contours.max() - contours.min()) / 50.0
        data2[data2 <= contours.min()] = contours.min() + eps
        data2[data2 >= contours.max()] = contours.max() - eps
        if data_on_elem:
            im = plt.tripcolor(x, y, elem2, facecolors=data2, cmap=cmap)
        else:
            im = plt.tricontourf(
                x, y, elem2, data2, levels=contours, cmap=cmap, extend=extend
            )
        if do_cbar:
            cbar = map.colorbar(im, "bottom", size="5%", pad="2%")

    # 		n=642155-1
    # 		n=83089-1
    # 		plt.plot(x[n-1], y[n-1], markersize=10, marker='o')
    elif oce == "np":
        data2 = np.copy(data2)
        elem2 = mesh.elem  # [mesh.no_cyclic_elem,:]
        d = data2[elem2].mean(axis=1)
        k = [i for (i, val) in enumerate(d) if not np.isnan(val)]
        elem2 = elem2[k, :]
        print("ftriplot, number of dummy points:", len(d) - len(k))
        map = Basemap(projection="nplaea", boundinglat=45, lon_0=0, resolution="l")
        x, y = map(mesh.x2, mesh.y2)
        map.drawcoastlines()
        map.drawparallels(np.arange(-80.0, 81.0, 20.0), labels=plabels)
        map.drawmeridians(np.arange(-180.0, 181.0, 20.0), labels=mlabels)  # [0,1,0,0]
        map.drawmapboundary(fill_color="0.9")
        map.fillcontinents(color=".7", lake_color=".7")
        # data2[data2>900]=np.nan
        eps = (contours.max() - contours.min()) / 100.0
        data2[data2 <= contours.min()] = contours.min() + eps
        data2[data2 >= contours.max()] = contours.max() - eps
        im = plt.tricontourf(
            x, y, elem2, data2, levels=contours, cmap=cmap, extend=extend
        )
        if do_cbar:
            cbar = map.colorbar(im, "bottom", size="5%", pad="2%")
    elif oce == "sp":
        data2 = np.copy(data2)
        elem2 = mesh.elem  # [mesh.no_cyclic_elem,:]
        d = data2[elem2].mean(axis=1)
        k = [i for (i, val) in enumerate(d) if not np.isnan(val)]
        elem2 = elem2[k, :]
        print("ftriplot, number of dummy points:", len(d) - len(k))
        map = Basemap(projection="splaea", boundinglat=-20, lon_0=180, resolution="l")
        x, y = map(mesh.x2, mesh.y2)
        map.drawcoastlines()
        map.drawparallels(np.arange(-80.0, 81.0, 20.0), labels=plabels)
        map.drawmeridians(np.arange(-180.0, 181.0, 20.0), labels=mlabels)
        map.drawmapboundary(fill_color="0.9")
        map.fillcontinents(color=".7", lake_color=".7")
        # data2[data2>900]=np.nan
        eps = (contours.max() - contours.min()) / 100.0
        data2[data2 <= contours.min()] = contours.min() + eps
        data2[data2 >= contours.max()] = contours.max() - eps
        im = plt.tricontourf(
            x, y, elem2, data2, levels=contours, cmap=cmap, extend=extend
        )
        if do_cbar:
            cbar = map.colorbar(im, "bottom", size="5%", pad="2%")
    return (im, map, cbar if (do_cbar) else False)


def wplot_xy(xx, yy, zz, contours, cmap=[], do_cbar=True, oce="global"):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    from matplotlib.colors import LinearSegmentedColormap

    if cmap == []:
        cmap = plt.cm.jet
    eps = (contours.max() - contours.min()) / 100.0
    zz[zz <= contours.min()] = contours.min() + eps
    zz[zz >= contours.max()] = contours.max() - eps

    if oce == "global":

        map = Basemap(projection="robin", lon_0=0, llcrnrlon=-180.0, urcrnrlon=180.0)
        xxx, yyy = map(xx, yy)

        map.drawmapboundary(fill_color="0.9")
        map.drawcoastlines()
        map.fillcontinents(color=".7", lake_color=".7")
        map.drawparallels(np.arange(-90, 90, 45), labels=[1, 0, 0, 0])
        map.drawmeridians([-120.0, 0.0, 120.0], labels=[0, 0, 0, 1])
        im = plt.contourf(xxx, yyy, zz, levels=contours, cmap=cmap, extend="both")
        if do_cbar:
            cbar = map.colorbar(im, "bottom", size="5%", pad="2%")
            return (im, map, cbar)
        else:
            return (im, map)
    elif oce == "np":
        map = Basemap(projection="nplaea", boundinglat=45, lon_0=0, resolution="l")
        xxx, yyy = map(xx, yy)

        map.drawmapboundary(fill_color="0.9")
        map.drawcoastlines()
        map.fillcontinents(color=".7", lake_color=".7")
        map.drawparallels(np.arange(-80.0, 81.0, 20.0), labels=[0, 0, 0, 0])
        map.drawmeridians(
            np.arange(-180.0, 181.0, 20.0), labels=[0, 0, 0, 0]
        )  # [0,1,0,0]
        im = plt.contourf(xxx, yyy, zz, levels=contours, cmap=cmap, extend="both")
        if do_cbar:
            cbar = map.colorbar(im, "bottom", size="5%", pad="2%")
            return (im, map, cbar)
        else:
            return (im, map)
    elif oce == "sp":
        map = Basemap(projection="splaea", boundinglat=-20, lon_0=180, resolution="l")
        xxx, yyy = map(xx, yy)

        map.drawmapboundary(fill_color="0.9")
        map.drawcoastlines()
        map.fillcontinents(color=".7", lake_color=".7")
        map.drawparallels(np.arange(-80.0, 81.0, 20.0), labels=[0, 0, 0, 0])
        map.drawmeridians(np.arange(-180.0, 181.0, 20.0), labels=[0, 0, 0, 0])
        im = plt.contourf(xxx, yyy, zz, levels=contours, cmap=cmap, extend="both")
        if do_cbar:
            cbar = map.colorbar(im, "bottom", size="5%", pad="2%")
            return (im, map, cbar)
        else:
            return (im, map)


def wplot_yz(y, z, v, contours, cmap=[]):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if cmap == []:
        cmap = plt.cm.jet

    im = plt.contourf(y, z, v, levels=contours, cmap=cmap, extend="both")
    cbar = plt.colorbar(orientation="horizontal")
    plt.grid()
    return (im, cbar)


def movingaverage(interval, window_size):
    import numpy as np

    window = np.ones(int(window_size)) / float(window_size)
    ret = list(interval)
    for i in range(window_size):
        ret = ret + [ret[-1]]
    ret = np.convolve(np.array(ret), window, "valid")
    return ret


def plot(
    mesh,
    data,
    cmap=None,
    influence=80000,
    box=[-180, 180, -89, 90],
    res=[360, 180],
    interp="nn",
    mapproj="merc",
    levels=None,
    ptype="cf",
    units=None,
    figsize=(10, 10),
    rowscol=(1, 1),
    titles=None,
    distances_path=None,
    inds_path=None,
    qhull_path=None,
    basepath=None,
):
    """
    Plots interpolated 2d field on the map.

    Parameters
    ----------
    mesh: mesh object
        FESOM2 mesh object
    data: np.array or list of np.arrays
        FESOM 2 data on nodes (for u,v,u_ice and v_ice one have to first interpolate from elements to nodes).
        Can be ether one np.ndarray or list of np.ndarrays.
    cmap: str
        Name of the colormap from cmocean package or from the standard matplotlib set.
        By default `Spectral_r` will be used.
    influence: float
        Radius of influence for interpolation, in meters.
    box: list
        Map boundaries in -180 180 -90 90 format that will be used for interpolation (default [-180 180 -89 90]).
    res: list
        Number of points along each axis that will be used for interpolation (for lon and lat),
        default [360, 180].
    interp: str
        Interpolation method. Options are 'nn' (nearest neighbor), 'idist' (inverce distance), "linear" and "cubic".
    mapproj: str
        Map projection. Options are Mercator (merc), Plate Carree (pc),
        North Polar Stereo (np), South Polar Stereo (sp),  Robinson (rob)
    levels: list
        Levels for contour plot in format min max numberOfLevels.
        If not provided min/max values from data will be used with 40 levels.
    ptype: str
        Plot type. Options are contourf (\'cf\') and pcolormesh (\'pcm\')
    units: str
        Units for color bar.
    figsize: tuple
        figure size in inches
    rowscol: tuple
        number of rows and columns.
    titles: str or list
        Title of the plot (if string) or subplots (if list of strings)
    distances_path : string
        Path to the file with distances. If not provided and dumpfile=True, it will be created.
    inds_path : string
        Path to the file with inds. If not provided and dumpfile=True, it will be created.
    qhull_path : str
         Path to the file with qhull (needed for linear and cubic interpolations). If not provided and dumpfile=True, it will be created.
    basepath: str
        path where to store additional interpolation files. If None (default),
        the path of the mesh will be used.
    """
    if not isinstance(data, list):
        data = [data]
    if titles:
        if not isinstance(titles, list):
            titles = [titles]
        if len(titles) != len(data):
            raise ValueError(
                "The number of titles do not match the number of data fields, please adjust titles (or put to None)")

    if (rowscol[0] * rowscol[1]) < len(data):
        raise ValueError(
            "Number of rows*columns is smaller than number of data fields, please adjust rowcol.")

    if cmap:
        if isinstance(cmap, (matplotlib.colors.Colormap)):
            colormap = cmap
        elif cmap in cmo.cmapnames:
            colormap = cmo.cmap_d[cmap]
        elif cmap in plt.cm.datad:
            colormap = plt.get_cmap(cmap)
        else:
            raise ValueError(
                "Get unrecognised name for the colormap `{}`. Colormaps should be from standard matplotlib set of from cmocean package.".format(
                    cmap
                )
            )
    else:
        colormap = plt.get_cmap("Spectral_r")

    radius_of_influence = influence

    left, right, down, up = box
    lonNumber, latNumber = res

    # flf = Dataset(ifile)
    lonreg = np.linspace(left, right, lonNumber)
    latreg = np.linspace(down, up, latNumber)
    lonreg2, latreg2 = np.meshgrid(lonreg, latreg)

    interpolated = []
    for datainstance in data:

        if interp == "nn":
            ofesom = fesom2regular(
                datainstance,
                mesh,
                lonreg2,
                latreg2,
                distances_path=distances_path,
                inds_path=inds_path,
                radius_of_influence=radius_of_influence,
                basepath=basepath,
            )
            interpolated.append(ofesom)
        elif interp == "idist":
            ofesom = fesom2regular(
                datainstance,
                mesh,
                lonreg2,
                latreg2,
                distances_path=distances_path,
                inds_path=inds_path,
                radius_of_influence=radius_of_influence,
                how="idist",
                k=5,
                basepath=basepath,
            )
            interpolated.append(ofesom)
        elif interp == "linear":
            ofesom = fesom2regular(
                datainstance,
                mesh,
                lonreg2,
                latreg2,
                how="linear",
                qhull_path=qhull_path,
                basepath=basepath,
            )
            interpolated.append(ofesom)
        elif interp == "cubic":
            ofesom = fesom2regular(
                datainstance, mesh, lonreg2, latreg2, basepath=basepath, how="cubic"
            )
            interpolated.append(ofesom)

    # nearth = cfeature.NaturalEarthFeature("physical", "ocean", "50m")
    # main_geom = [contour for contour in nearth.geometries()][0]

    # mask = shapely.vectorized.contains(main_geom, lonreg2, latreg2)
    # m2 = np.where(((lonreg2 == -180.0) & (latreg2 > 71.5)), True, mask)
    # m2 = np.where(
    #     ((lonreg2 == -180.0) & (latreg2 < 70.95) & (latreg2 > 68.96)), True, m2
    # )
    # m2 = np.where(((lonreg2 == -180.0) & (latreg2 < 65.33)), True, m2)

    m2 = mask_ne(lonreg2, latreg2)

    #     m2 = np.where(((lonreg2 == 180.)&(latreg2>71.5)), True, m2)
    #     m2 = np.where(((lonreg2 == 180.)&(latreg2<70.95)&(latreg2>68.96)), True, m2)
    #     m2 = np.where(((lonreg2 == 180.)&(latreg2<65.33)), True, m2)

    for i, interpolated_instance in enumerate(interpolated):
        interpolated[i] = np.ma.masked_where(m2, interpolated[i])
        interpolated[i] = np.ma.masked_equal(interpolated[i], 0)

    if mapproj == "merc":
        fig, ax = plt.subplots(
            rowscol[0],
            rowscol[1],
            subplot_kw=dict(projection=ccrs.Mercator()),
            constrained_layout=True,
            figsize=figsize,
        )
    elif mapproj == "pc":
        fig, ax = plt.subplots(
            rowscol[0],
            rowscol[1],
            subplot_kw=dict(projection=ccrs.PlateCarree()),
            constrained_layout=True,
            figsize=figsize,
        )
    elif mapproj == "np":
        fig, ax = plt.subplots(
            rowscol[0],
            rowscol[1],
            subplot_kw=dict(projection=ccrs.NorthPolarStereo()),
            constrained_layout=True,
            figsize=figsize,
        )
    elif mapproj == "sp":
        fig, ax = plt.subplots(
            rowscol[0],
            rowscol[1],
            subplot_kw=dict(projection=ccrs.SouthPolarStereo()),
            constrained_layout=True,
            figsize=figsize,
        )
    elif mapproj == "rob":
        fig, ax = plt.subplots(
            rowscol[0],
            rowscol[1],
            subplot_kw=dict(projection=ccrs.Robinson()),
            constrained_layout=True,
            figsize=figsize,
        )
    if isinstance(ax, np.ndarray):
        ax = ax.flatten()
    else:
        ax = [ax]

    for ind, data_int in enumerate(interpolated):
        ax[ind].set_extent([left, right, down, up], crs=ccrs.PlateCarree())
        if levels:
            mmin, mmax, nnum = levels
            nnum = int(nnum)
        else:
            mmin = np.nanmin(data_int)
            mmax = np.nanmax(data_int)
            nnum = 40
        data_levels = np.linspace(mmin, mmax, nnum)
        if ptype == "cf":
            data_int_cyc, lon_cyc = add_cyclic_point(data_int, coord=lonreg)
            image = ax[ind].contourf(
                lon_cyc,
                latreg,
                data_int_cyc,
                levels=data_levels,
                transform=ccrs.PlateCarree(),
                cmap=colormap,
                extend="both",
            )
        elif ptype == "pcm":
            data_int_cyc, lon_cyc = add_cyclic_point(data_int, coord=lonreg)
            image = ax[ind].pcolormesh(
                lon_cyc,
                latreg,
                data_int_cyc,
                vmin=mmin,
                vmax=mmax,
                transform=ccrs.PlateCarree(),
                cmap=colormap,
            )
        else:
            raise ValueError("Inknown plot type {}".format(ptype))

        # ax.coastlines(resolution = '50m',lw=0.5)
        ax[ind].add_feature(
            cfeature.GSHHSFeature(levels=[1], scale="low", facecolor="lightgray")
        )
        if titles:
            ax[ind].set_title(titles.pop(0), size=20)

    for delind in range(ind + 1, len(ax)):
        fig.delaxes(ax[delind])

    cb = fig.colorbar(image, orientation="horizontal", ax=ax, pad=0.01, shrink=0.9)

    cb.ax.tick_params(labelsize=15)

    if units:
        cb.set_label(units, size=20)
    else:
        pass

    return ax


def plot_transect_map(
    lon_start, lat_start, lon_end, lat_end, mesh, npoints=30, view="w", stock_img=False
):
    # plt.figure(figsize=(10,10))
    lonlat = transect_get_lonlat(
        lon_start, lat_start, lon_end, lat_end, npoints=npoints
    )
    nodes = transect_get_nodes(lonlat, mesh)
    # dist   = transect_get_distance(lonlat)

    if view == "w":
        ax = plt.subplot(111, projection=ccrs.Mercator(central_longitude=0))
        ax.set_extent([180, -180, -80, 90], crs=ccrs.PlateCarree())
    elif view == "np":
        ax = plt.subplot(111, projection=ccrs.NorthPolarStereo(central_longitude=0))
        ax.set_extent([180, -180, 60, 90], crs=ccrs.PlateCarree())
    elif view == "sp":
        ax = plt.subplot(111, projection=ccrs.SouthPolarStereo(central_longitude=0))
        ax.set_extent([180, -180, -90, -50], crs=ccrs.PlateCarree())
    else:
        raise ValueError(
            'The "{}" is not recognized as valid view option.'.format(view)
        )

    ax.scatter(lonlat[:, 0], lonlat[:, 1], s=30, c="b", transform=ccrs.PlateCarree())
    ax.scatter(
        mesh.x2[nodes], mesh.y2[nodes], s=30, c="r", transform=ccrs.PlateCarree()
    )
    if stock_img == True:
        ax.stock_img()
    ax.coastlines(resolution="50m")
    return ax


def plot_transect(
    data3d,
    mesh,
    lon_start,
    lat_start,
    lon_end,
    lat_end,
    npoints=30,
    maxdepth=1000,
    label="$^{\circ}$C",
    title="",
    levels=None,
    cmap=cm.Spectral_r,
    ax=None,
    dist=None,
    nodes=None,
    ncols=2,
    figsize=None,
    transect_data=[],
    max_distance=1e6,
):

    depth_index = ind_for_depth(maxdepth, mesh)
    if not isinstance(data3d, list):
        if ax is None:
            ax = plt.gca()
            oneplot = True
        else:
            oneplot = False
        if (type(dist) is np.ndarray) and (type(nodes) is np.ndarray):
            if not (type(transect_data) is np.ma.core.MaskedArray):
                lonlat = transect_get_lonlat(
                    lon_start, lat_start, lon_end, lat_end, npoints=npoints
                )
                mask2d = transect_get_mask(nodes, mesh, lonlat, max_distance)
                transect_data = transect_get_data(data3d, nodes, mask2d)
        else:
            lonlat = transect_get_lonlat(
                lon_start, lat_start, lon_end, lat_end, npoints=npoints
            )
            nodes = transect_get_nodes(lonlat, mesh)
            dist = transect_get_distance(lonlat)
            # profile = transect_get_profile(nodes, mesh)
            if not (type(transect_data) is np.ma.core.MaskedArray):
                mask2d = transect_get_mask(nodes, mesh, lonlat, max_distance)
                transect_data = transect_get_data(data3d, nodes, mask2d)

        image = ax.contourf(
            dist,
            np.abs(mesh.zlev[:depth_index]),
            transect_data[:, :depth_index].T,
            levels=levels,
            cmap=cmap,
            extend="both",
        )
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("km")
        ax.set_ylabel("m")

        if oneplot:
            cb = plt.colorbar(image)
            cb.set_label(label)

        return image
    else:
        ncols = float(ncols)
        nplots = len(data3d)
        nrows = math.ceil(nplots / ncols)
        ncols = int(ncols)
        nrows = int(nrows)
        nplot = 1

        if not figsize:
            figsize = (8 * ncols, 2 * nrows * ncols)
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
        ax = ax.flatten()
        for ind, data in enumerate(data3d):
            if (type(dist) is np.ndarray) and (type(nodes) is np.ndarray):
                transect_data = transect_get_data(data, nodes)
            else:
                lonlat = transect_get_lonlat(
                    lon_start, lat_start, lon_end, lat_end, npoints=npoints
                )
                nodes = transect_get_nodes(lonlat, mesh)
                dist = transect_get_distance(lonlat)
                # profile = transect_get_profile(nodes, mesh)
                mask2d = transect_get_mask(nodes, mesh, lonlat, max_distance)
                transect_data = transect_get_data(data3d, nodes, max_distance)

            image = ax[ind].contourf(
                dist,
                np.abs(mesh.zlev[:depth_index]),
                transect_data[:, :depth_index].T,
                levels=levels,
                cmap=cmap,
                extend="both",
            )
            ax[ind].invert_yaxis()
            if not isinstance(title, list):
                ax[ind].set_title(title)
            else:
                ax[ind].set_title(title[ind])
            ax[ind].set_xlabel("km")
            ax[ind].set_ylabel("m")

            cb = fig.colorbar(image, orientation="horizontal", ax=ax[ind], pad=0.11)
            cb.set_label(label)
        for delind in range(ind + 1, len(ax)):

            fig.delaxes(ax[delind])

        fig.tight_layout()
