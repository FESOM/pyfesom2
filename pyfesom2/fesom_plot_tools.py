# -*- coding: utf-8 -*-
#
# This file is part of pyfesom2
# Original code by Dmitry Sidorenko, Qiang Wang, Sergey Danilov and Patrick Scholz
#

import sys, os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from .regriding import fesom2regular
from netCDF4 import Dataset, MFDataset, num2date
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point
except ImportError:
    print("Cartopy is not installed, plotting is not available.")
from cmocean import cm as cmo
from matplotlib import cm

import xarray as xr
import shapely.vectorized
import joblib
from .transect import *
from .ut import mask_ne, cut_region, get_cmap, get_no_cyclic
from matplotlib import ticker
import math

sfmt = ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((-3, 4))


def create_proj_figure(mapproj, rowscol, figsize):
    """ Create figure and axis with cartopy projection.

    Parameters
    ----------
    mapproj: str
        name of the projection:
            merc: Mercator
            pc: PlateCarree (default)
            np: NorthPolarStereo
            sp: SouthPolarStereo
            rob: Robinson
    rowcol: (int, int)
        number of rows and columns of the figure.
    figsize: (float, float)
        width, height in inches.

    Returns
    -------
    fig, ax

    """
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
    else:
        raise ValueError(f"Projection {mapproj} is not supported.")
    return fig, ax


def get_plot_levels(levels, data, lev_to_data=False):
    """Returns levels for the plot.

    Parameters
    ----------
    levels: list, numpy array
        Can be list or numpy array with three or more elements.
        If only three elements provided, they will b einterpereted as min, max, number of levels.
        If more elements provided, they will be used directly.
    data: numpy array of xarray
        Data, that should be plotted with this levels.
    lev_to_data: bool
        Switch to correct the levels to the actual data range. 
        This is needed for safe plotting on triangular grid with cartopy.

    Returns
    -------
    data_levels: numpy array
        resulted levels.
        
    """
    if levels:
        if len(levels) == 3:
            mmin, mmax, nnum = levels
            if lev_to_data:
                mmin, mmax = levels_to_data(mmin, mmax, data)
            nnum = int(nnum)
            data_levels = np.linspace(mmin, mmax, nnum)
        elif len(levels) < 3:
            raise ValueError(
                "Levels can be the list or numpy array with three or more elements."
            )
        else:
            data_levels = np.array(levels)
    else:
        mmin = np.nanmin(data)
        mmax = np.nanmax(data)
        nnum = 40
        data_levels = np.linspace(mmin, mmax, nnum)
    return data_levels


def levels_to_data(mmin, mmax, data):
    """Correct the levels to the actual data range.

    This is needed to make cartopy happy. 
    Cartopy can't plot on triangular mesh when the color
    range is larger than the data range.
    """
    # this is needed to make cartopy happy
    mmin_d = np.nanmin(data)
    mmax_d = np.nanmax(data)
    if mmin < mmin_d:
        mmin = mmin_d
        print("minimum level changed to make cartopy happy")
    if mmax > mmax_d:
        mmax = mmax_d
        print("maximum level changed to make cartopy happy")
    return mmin, mmax


def plot(
    mesh,
    data,
    cmap=None,
    influence=80000,
    box=[-180, 180, -89, 90],
    res=[360, 180],
    interp="nn",
    mapproj="pc",
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
                "The number of titles do not match the number of data fields, please adjust titles (or put to None)"
            )

    if (rowscol[0] * rowscol[1]) < len(data):
        raise ValueError(
            "Number of rows*columns is smaller than number of data fields, please adjust rowscol."
        )

    if cmap:
        if isinstance(cmap, (mpl.colors.Colormap)):
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
            titles = titles.copy()
            ax[ind].set_title(titles.pop(0), size=20)

    for delind in range(ind + 1, len(ax)):
        fig.delaxes(ax[delind])

    cb = fig.colorbar(
        image, orientation="horizontal", ax=ax, pad=0.01, shrink=0.9, format=sfmt
    )

    cb.ax.tick_params(labelsize=15)

    if units:
        cb.set_label(units, size=20)
    else:
        pass

    return ax


def plot_transect_map(lonlat, mesh, view="w", stock_img=False):
    """Plot map of the transect.
    
    Parameters
    ----------
    lonlat : np.array
        2 dimentional np. array that contains longitudea and latitudes.
        Can be constructed from vectors as lonlat = np.vstack((lon, lat))
    mesh: mesj object
    view: str
        Projection to use for the map:
        w - global (Mercator)
        np - North Polar Stereo
        sp - South Polar Stereo
    stock_imd: bool
        Show stock backgroung image. Usually makes things slower.
    
    Returns
    -------
    ax: cartopy axis object
    
    """

    nodes = transect_get_nodes(lonlat, mesh)

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

    ax.scatter(lonlat[0, :], lonlat[1, :], s=30, c="b", transform=ccrs.PlateCarree())
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
    lonlat,
    maxdepth=1000,
    label=r"$^{\circ}$C",
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
    facecolor="lightgray",
    fontsize=12,
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
                mask2d = transect_get_mask(nodes, mesh, lonlat, max_distance)
                transect_data = transect_get_data(data3d, nodes, mask2d)
        else:
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
        ax.set_title(title, size=fontsize)
        ax.set_xlabel("km", size=fontsize)
        ax.set_ylabel("m", size=fontsize)
        ax.set_facecolor(facecolor)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)

        if oneplot:
            cb = plt.colorbar(image, format=sfmt)
            cb.set_label(label, size=fontsize)
            cb.ax.tick_params(labelsize=fontsize)
            cb.ax.yaxis.get_offset_text().set_fontsize(fontsize)

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
            if (type(data) is np.ndarray) and (type(nodes) is np.ndarray):
                if not (type(transect_data) is np.ma.core.MaskedArray):
                    mask2d = transect_get_mask(nodes, mesh, lonlat, max_distance)
                    transect_data = transect_get_data(data, nodes, mask2d)
            else:
                nodes = transect_get_nodes(lonlat, mesh)
                dist = transect_get_distance(lonlat)
                # profile = transect_get_profile(nodes, mesh)
                if not (type(transect_data) is np.ma.core.MaskedArray):
                    mask2d = transect_get_mask(nodes, mesh, lonlat, max_distance)
                    transect_data = transect_get_data(data, nodes, mask2d)

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
                ax[ind].set_title(title, size=fontsize)
            else:
                ax[ind].set_title(title[ind], size=fontsize)

            ax[ind].set_xlabel("km", size=fontsize)
            ax[ind].set_ylabel("m", size=fontsize)
            ax[ind].set_facecolor(facecolor)
            ax[ind].tick_params(axis="both", which="major", labelsize=fontsize)

            cb = fig.colorbar(
                image, orientation="horizontal", ax=ax[ind], pad=0.11, format=sfmt
            )
            cb.set_label(label, size=fontsize)
            cb.ax.tick_params(labelsize=fontsize)
            cb.ax.xaxis.get_offset_text().set_fontsize(fontsize)

        for delind in range(ind + 1, len(ax)):

            fig.delaxes(ax[delind])

        fig.tight_layout()


def hofm_plot_one(
    mesh,
    data,
    xvals,
    levels=None,
    maxdepth=1000,
    label=r"$^{\circ}$C",
    title="",
    cmap=cm.Spectral_r,
    ax=None,
    facecolor="lightgray",
    fontsize=12,
    xlabel="Time",
):
    depth_index = ind_for_depth(maxdepth, mesh)

    if ax is None:
        ax = plt.gca()
        oneplot = True
    else:
        oneplot = False

    image = ax.contourf(
        xvals,
        np.abs(mesh.zlev[:depth_index]),
        data[:, :depth_index].T,
        levels=levels,
        cmap=cmap,
        extend="both",
    )
    ax.invert_yaxis()
    ax.set_title(title, size=fontsize)
    ax.set_xlabel(xlabel, size=fontsize)
    ax.set_ylabel("Depth, m", size=fontsize)
    ax.set_facecolor(facecolor)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)

    if oneplot:
        cb = plt.colorbar(image, format=sfmt)
        cb.set_label(label, size=fontsize)
        cb.ax.tick_params(labelsize=fontsize)
        cb.ax.yaxis.get_offset_text().set_fontsize(fontsize)

    return image


def hofm_plot_many(
    mesh,
    data,
    xvals,
    levels=None,
    maxdepth=1000,
    label=r"$^{\circ}$C",
    title="",
    cmap=cm.Spectral_r,
    ax=None,
    facecolor="lightgray",
    fontsize=12,
    ncols=2,
    figsize=None,
    xlabel="Time",
):
    depth_index = ind_for_depth(maxdepth, mesh)

    ncols = float(ncols)
    nplots = len(data)
    nrows = math.ceil(nplots / ncols)
    ncols = int(ncols)
    nrows = int(nrows)
    nplot = 1

    if not figsize:
        figsize = (8 * ncols, 2 * nrows * ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    ax = ax.flatten()

    for ind, data_one in enumerate(data):

        image = ax[ind].contourf(
            xvals,
            np.abs(mesh.zlev[:depth_index]),
            data_one[:, :depth_index].T,
            levels=levels,
            cmap=cmap,
            extend="both",
        )
        ax[ind].invert_yaxis()

        if not isinstance(title, list):
            ax[ind].set_title(title, size=fontsize)
        else:
            ax[ind].set_title(title[ind], size=fontsize)

        ax[ind].set_xlabel(xlabel, size=fontsize)
        ax[ind].set_ylabel("Depth, m", size=fontsize)
        ax[ind].set_facecolor(facecolor)
        ax[ind].tick_params(axis="both", which="major", labelsize=fontsize)

        cb = fig.colorbar(
            image, orientation="horizontal", ax=ax[ind], pad=0.11, format=sfmt
        )
        cb.set_label(label, size=fontsize)
        cb.ax.tick_params(labelsize=fontsize)
        cb.ax.xaxis.get_offset_text().set_fontsize(fontsize)

    for delind in range(ind + 1, len(ax)):

        fig.delaxes(ax[delind])

    fig.tight_layout()

    return fig


def hofm_plot(
    mesh,
    data,
    xvals=None,
    levels=None,
    maxdepth=1000,
    label=r"$^{\circ}$C",
    title="",
    cmap=cm.Spectral_r,
    ax=None,
    facecolor="lightgray",
    fontsize=12,
    ncols=2,
    figsize=None,
    xlabel="Time",
):
    """ Plot data on x (e.g. time, distance) / depth.

    Parameters:
    -----------
    mesh: mesh object
        pyfesom2 mesh object
    data: 2D xarray, nd array or list of them
        2D input data. Can be ether one or several (as a list)
        numpy arrays or xarray DataArrays. If list of arrays is
        provided, several plots will be ploted at once in a multipanel.
    xvals: nd array
        Values for the x axis (e.g. time, distance).
        Deduced automatically if `data` is xarray DataArray.
        Should be provided if `data` is nd array.
    levels:
        list of levels for contour plot
    maxdepth:
        maximum depth the plot will be limited to
    label:
        label for colorbar
    title: str or list
        Should be str if only one plot is expected.
        For multipanel plots should be the list of strings.
    cmap:
        matplotlib colormap instance
    ax: matplotlib ax
        Only for single plot. It can be inserted to other figure.
    facecolor: str
        Used to fill aread with NaNs.
        Should be the name of the color that matplotlib can understand.
    fontsize: int
        Font size of text elements (e.g. labeles)
    ncols: int
        Number of columns for multipanel plot.
    figsize: tuple
        ONLY works for multipanel plots.
        For single plots use plt.figure(figsize=(10, 10))
        before calling this function/
    xlabel:
        Label for x axis.
    """

    if not isinstance(data, list):
        if isinstance(data, xr.DataArray):
            xvals = data.time.data
        else:
            if xvals is None:
                raise ValueError(
                    "You provide np.array as an input, but did not provide xvals (e.g. time or distance)"
                )

        hofm_plot_one(
            mesh=mesh,
            data=data,
            xvals=xvals,
            levels=levels,
            maxdepth=maxdepth,
            label=label,
            title=title,
            cmap=cmap,
            ax=ax,
            facecolor=facecolor,
            fontsize=fontsize,
            xlabel=xlabel,
        )
    else:
        if isinstance(data[0], xr.DataArray):
            xvals = data[0].time.data
        else:
            if xvals is None:
                raise ValueError(
                    "You provide np.array as an input, but did not provide xvals (e.g. time or distance)"
                )

        hofm_plot_many(
            mesh=mesh,
            data=data,
            xvals=xvals,
            levels=levels,
            maxdepth=maxdepth,
            label=label,
            title=title,
            cmap=cmap,
            ax=ax,
            facecolor=facecolor,
            fontsize=fontsize,
            ncols=ncols,
            figsize=figsize,
            xlabel=xlabel,
        )


def tplot(
    mesh,
    data,
    cmap=None,
    box=[-180, 180, -80, 90],
    mapproj="pc",
    levels=None,
    ptype="cf",
    units=r"$^\circ$C",
    figsize=(10, 10),
    rowscol=(1, 1),
    titles=None,
    lw=0.01,
    fontsize=12,
):
    """Plots original field on the cartopy map using tricontourf or tripcolor.

    Parameters
    ----------
    mesh: mesh object
        FESOM2 mesh object
    data: np.array or list of np.arrays
        FESOM 2 data on nodes 
        (for u,v,u_ice and v_ice one have to first interpolate from elements to nodes (`tonodes` function)).
        Can be ether one np.ndarray or list of np.ndarrays.
    cmap: str
        Name of the colormap from cmocean package or from the standard matplotlib set.
        By default `Spectral_r` will be used.
    box: list
        Map boundaries in -180 180 -90 90 format that will be used for data selection and plotting (default [-180 180 -89 90]).
    mapproj: str
        Map projection. Options are Mercator (merc), Plate Carree (pc),
        North Polar Stereo (np), South Polar Stereo (sp),  Robinson (rob)
    levels: list
        Levels for contour plot in format (min, max, numberOfLevels). List with more than
        3 values will be interpreted as just a list of individual level values.
        If not provided min/max values from data will be used with 40 levels.
    ptype: str
        Plot type. Options are tricontourf (\'cf\') and tripcolor (\'tri\')
    units: str
        Units for color bar.
    figsize: tuple
        figure size in inches
    rowscol: tuple
        number of rows and columns.
    titles: str or list
        Title of the plot (if string) or subplots (if list of strings)
    fontsize: float
        Font size of some of the plot elements.
    """

    if not isinstance(data, list):
        data = [data]
    if titles:
        if not isinstance(titles, list):
            titles = [titles]
        if len(titles) != len(data):
            raise ValueError(
                "The number of titles do not match the number of data fields, please adjust titles (or put to None)"
            )

    if (rowscol[0] * rowscol[1]) < len(data):
        raise ValueError(
            "Number of rows*columns is smaller than number of data fields, please adjust rowscol."
        )

    colormap = get_cmap(cmap=cmap)

    fig, ax = create_proj_figure(mapproj, rowscol, figsize)
    if isinstance(ax, np.ndarray):
        ax = ax.flatten()
    else:
        ax = [ax]

    for ind, data_to_plot in enumerate(data):
        data_levels = get_plot_levels(levels, data_to_plot, lev_to_data=True)
        #     ax.set_global()
        ax[ind].set_extent(box, crs=ccrs.PlateCarree())

        if ptype == "tri":

            elem_no_nan = cut_region(mesh, box)
            no_cyclic_elem2 = get_no_cyclic(mesh, elem_no_nan)
            # masked values do not work in cartopy
            data_to_plot[data_to_plot == 0] = -99999
            image = ax[ind].tripcolor(
                mesh.x2,
                mesh.y2,
                elem_no_nan[no_cyclic_elem2],
                data_to_plot,
                transform=ccrs.PlateCarree(),
                cmap=colormap,
                vmin=data_levels[0],
                vmax=data_levels[-1],
                edgecolors="k",
                lw=lw,
                alpha=1,
            )
        elif ptype == "cf":
            elem_no_nan = cut_region(mesh, box)
            no_cyclic_elem2 = get_no_cyclic(mesh, elem_no_nan)
            # masked values do not work in cartopy
            data_to_plot[data_to_plot == 0] = -99999
            image = ax[ind].tricontourf(
                mesh.x2,
                mesh.y2,
                elem_no_nan[no_cyclic_elem2],
                data_to_plot,
                levels=data_levels,
                transform=ccrs.PlateCarree(),
                cmap=colormap,
            )
        else:
            raise ValueError(
                "Only `cf` (contourf) and `tri` (tripcolor) options are supported."
            )

        ax[ind].coastlines(lw=1.5, resolution="110m")

        if titles:
            titles = titles.copy()
            ax[ind].set_title(titles.pop(0), size=20)

    for delind in range(ind + 1, len(ax)):
        fig.delaxes(ax[delind])

    cb = fig.colorbar(
        image, orientation="horizontal", ax=ax, pad=0.01, shrink=0.9, format=sfmt
    )

    cb.ax.tick_params(labelsize=fontsize)

    if units:
        cb.set_label(units, size=fontsize)
    else:
        pass
