# -*- coding: utf-8 -*-
#
# This file is part of pyfesom2
# Original code by Dmitry Sidorenko, Nikolay Koldunov,
# Qiang Wang, Sergey Danilov and Patrick Scholz
#

import math
import os
import sys
from typing import List, Tuple, Union, Optional, Dict, Any

import joblib
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import shapely.vectorized
import xarray as xr
from cmocean import cm as cmo
from matplotlib import cm, ticker
from matplotlib.colors import LinearSegmentedColormap
from netCDF4 import Dataset, MFDataset, num2date

from .load_mesh_data import ind_for_depth
from .regridding import fesom2regular, tonodes
from .transect import transect_get_nodes
from .ut import cut_region, get_cmap, get_no_cyclic, mask_ne, vec_rotate_r2g

# Format for scalar formatter
sfmt = ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((-3, 4))

# Try importing cartopy, but don't fail if not available
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point
    CARTOPY_AVAILABLE = True
except ImportError:
    print("Cartopy is not installed, plotting is not available.")
    CARTOPY_AVAILABLE = False


def create_proj_figure(
    mapproj: str, 
    rowscol: Tuple[int, int], 
    figsize: Tuple[float, float]
) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """Create figure and axis with cartopy projection.

    Parameters
    ----------
    mapproj: str
        Name of the projection:
            merc: Mercator
            pc: PlateCarree (default)
            np: NorthPolarStereo
            sp: SouthPolarStereo
            rob: Robinson
    rowscol: Tuple[int, int]
        Number of rows and columns of the figure.
    figsize: Tuple[float, float]
        Width, height in inches.

    Returns
    -------
    Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]
        Figure and axis objects
    """
    if not CARTOPY_AVAILABLE:
        raise ImportError("Cartopy is required for projection plots")
        
    projections = {
        "merc": ccrs.Mercator(),
        "pc": ccrs.PlateCarree(),
        "np": ccrs.NorthPolarStereo(),
        "sp": ccrs.SouthPolarStereo(),
        "rob": ccrs.Robinson()
    }
    
    if mapproj not in projections:
        raise ValueError(f"Projection {mapproj} is not supported. Choose from: {', '.join(projections.keys())}")
    
    fig, ax = plt.subplots(
        rowscol[0],
        rowscol[1],
        subplot_kw=dict(projection=projections[mapproj]),
        constrained_layout=True,
        figsize=figsize,
    )
    
    return fig, ax


def get_plot_levels(
    levels: Optional[List], 
    data: Union[np.ndarray, xr.DataArray], 
    lev_to_data: bool = False
) -> np.ndarray:
    """Returns levels for the plot.

    Parameters
    ----------
    levels: Optional[List]
        Can be list or numpy array with three or more elements.
        If only three elements provided, they will be interpreted as min, max, number of levels.
        If more elements provided, they will be used directly.
    data: Union[np.ndarray, xr.DataArray]
        Data that should be plotted with these levels.
    lev_to_data: bool
        Switch to correct the levels to the actual data range.
        This is needed for safe plotting on triangular grid with cartopy.

    Returns
    -------
    np.ndarray
        Resulting levels.
    """
    if levels is not None:
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


def levels_to_data(
    mmin: float, 
    mmax: float, 
    data: Union[np.ndarray, xr.DataArray]
) -> Tuple[float, float]:
    """Correct the levels to the actual data range.

    This is needed to make cartopy happy.
    Cartopy can't plot on triangular mesh when the color
    range is larger than the data range.
    
    Parameters
    ----------
    mmin: float
        Minimum value for levels
    mmax: float
        Maximum value for levels
    data: Union[np.ndarray, xr.DataArray]
        Data to be plotted
        
    Returns
    -------
    Tuple[float, float]
        Corrected min and max values
    """
    mmin_d = np.nanmin(data)
    mmax_d = np.nanmax(data)
    
    if mmin < mmin_d:
        mmin = mmin_d
        print("Minimum level changed to make cartopy happy")
    if mmax > mmax_d:
        mmax = mmax_d
        print("Maximum level changed to make cartopy happy")
    
    return mmin, mmax


def interpolate_for_plot(
    data: List[np.ndarray],
    mesh: Any,
    lonreg2: np.ndarray,
    latreg2: np.ndarray,
    interp: str = "nn",
    distances_path: Optional[str] = None,
    inds_path: Optional[str] = None,
    radius_of_influence: Optional[float] = None,
    basepath: Optional[str] = None,
    qhull_path: Optional[str] = None,
) -> List[np.ndarray]:
    """Interpolate for the plot.

    Parameters
    ----------
    data: List[np.ndarray]
        FESOM 2 data on nodes (for u,v,u_ice and v_ice one has to first interpolate from elements to nodes).
        Can be either one np.ndarray or list of np.ndarrays.
    mesh: Any
        FESOM2 mesh object
    lonreg2: np.ndarray
        Longitudes of the regular grid.
    latreg2: np.ndarray
        Latitudes of the regular grid.
    interp: str
        Interpolation method. Options are 'nn' (nearest neighbor), 'idist' (inverse distance), "linear" and "cubic".
    distances_path: Optional[str]
        Path to the file with distances. If not provided and dumpfile=True, it will be created.
    inds_path: Optional[str]
        Path to the file with inds. If not provided and dumpfile=True, it will be created.
    radius_of_influence: Optional[float]
        Radius for nearest neighbor interpolation.
    basepath: Optional[str]
        Path where to store additional interpolation files. If None (default),
        the path of the mesh will be used.
    qhull_path: Optional[str]
        Path to the file with qhull (needed for linear and cubic interpolations). 
        If not provided and dumpfile=True, it will be created.

    Returns
    -------
    List[np.ndarray]
        Interpolated data on regular grid
    """
    interpolation_methods = {
        "nn": lambda d: fesom2regular(
            d, mesh, lonreg2, latreg2, 
            distances_path=distances_path,
            inds_path=inds_path,
            radius_of_influence=radius_of_influence,
            basepath=basepath
        ),
        "idist": lambda d: fesom2regular(
            d, mesh, lonreg2, latreg2,
            distances_path=distances_path,
            inds_path=inds_path,
            radius_of_influence=radius_of_influence,
            how="idist", k=5,
            basepath=basepath
        ),
        "linear": lambda d: fesom2regular(
            d, mesh, lonreg2, latreg2,
            how="linear",
            qhull_path=qhull_path,
            basepath=basepath
        ),
        "cubic": lambda d: fesom2regular(
            d, mesh, lonreg2, latreg2,
            how="cubic",
            basepath=basepath
        )
    }
    
    if interp not in interpolation_methods:
        raise ValueError(f"Interpolation method '{interp}' not supported. Choose from: {', '.join(interpolation_methods.keys())}")
    
    interpolated = []
    for datainstance in data:
        ofesom = interpolation_methods[interp](datainstance)
        interpolated.append(ofesom)
    
    return interpolated




def get_vector_forplot(
    u: np.ndarray,
    v: np.ndarray,
    mesh: Any,
    box: List[float] = [-180, 180, -89, 90],
    res: List[int] = [360, 180],
    influence: float = 80000,
    lonreg2: Optional[np.ndarray] = None,
    latreg2: Optional[np.ndarray] = None,
    no_pi_mask: bool = False,
    sea_ice: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare vector data for plotting.

    Parameters
    ----------
    u: np.ndarray
        u component of the vector field
    v: np.ndarray
        v component of the vector field
    mesh: Any
        FESOM2 mesh object
    box: List[float]
        Map boundaries in -180 180 -90 90 format
    res: List[int]
        Number of points along each axis that will be used for interpolation
    influence: float
        Radius of influence for interpolation, in meters
    lonreg2: Optional[np.ndarray]
        Longitudes of the regular grid (will be generated if not provided)
    latreg2: Optional[np.ndarray]
        Latitudes of the regular grid (will be generated if not provided)
    no_pi_mask: bool
        Whether to mask points on land
    sea_ice: bool
        Whether the data is sea ice velocity

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Interpolated u and v components and coordinates
    """
    if len(u.shape) > 1:
        raise ValueError(
            "You are trying to use 2D variable. Only 1D variables on elements can be used."
        )
    if len(v.shape) > 1:
        raise ValueError(
            "You are trying to use 2D variable. Only 1D variables on elements can be used."
        )
    
    # Convert from elements to nodes if not sea ice
    if not sea_ice:
        u_nodes = tonodes(u, mesh)
        v_nodes = tonodes(v, mesh)
    else:
        u_nodes = u
        v_nodes = v

    # Get box boundaries
    left, right, down, up = box

    # Create grid if not provided
    if (lonreg2 is None) and (latreg2 is None):
        lonNumber, latNumber = res
        lonreg = np.linspace(left, right, lonNumber)
        latreg = np.linspace(down, up, latNumber)
        lonreg2, latreg2 = np.meshgrid(lonreg, latreg)

    # Rotate vectors
    u_rot, v_rot = vec_rotate_r2g(
        50, 15, -90, mesh.x2, mesh.y2, u_nodes, v_nodes, flag=1
    )

    # Interpolate rotated vectors
    u_int = interpolate_for_plot(
        [u_rot],
        mesh,
        lonreg2,
        latreg2,
        interp="nn",
        radius_of_influence=influence
    )
    v_int = interpolate_for_plot(
        [v_rot],
        mesh,
        lonreg2,
        latreg2,
        interp="nn",
        radius_of_influence=influence
    )
    
    # Apply masks
    m2 = mask_ne(lonreg2, latreg2)
    u_int = u_int[0]
    v_int = v_int[0]

    if not no_pi_mask:
        u_int = np.ma.masked_where(m2, u_int)
    u_int = np.ma.masked_equal(u_int, 0)

    if not no_pi_mask:
        v_int = np.ma.masked_where(m2, v_int)
    v_int = np.ma.masked_equal(v_int, 0)

    return u_int, v_int, lonreg2, latreg2


def plot(
    mesh: Any,
    data: Union[np.ndarray, List[np.ndarray]],
    cmap: Optional[str] = None,
    influence: float = 80000,
    box: List[float] = [-180, 180, -89, 90],
    res: List[int] = [360, 180],
    interp: str = "nn",
    mapproj: str = "pc",
    levels: Optional[List] = None,
    ptype: str = "cf",
    units: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    rowscol: Tuple[int, int] = (1, 1),
    titles: Optional[Union[str, List[str]]] = None,
    distances_path: Optional[str] = None,
    inds_path: Optional[str] = None,
    qhull_path: Optional[str] = None,
    basepath: Optional[str] = None,
    interpolated_data: Optional[np.ndarray] = None,
    lonreg: Optional[np.ndarray] = None,
    latreg: Optional[np.ndarray] = None,
    no_pi_mask: bool = False,
) -> Union[plt.Axes, List[plt.Axes]]:
    """
    Plots interpolated 2d field on the map.

    Parameters
    ----------
    mesh: Any
        FESOM2 mesh object
    data: Union[np.ndarray, List[np.ndarray]]
        FESOM 2 data on nodes (for u,v,u_ice and v_ice one has to first interpolate from elements to nodes).
        Can be either one np.ndarray or list of np.ndarrays.
    cmap: Optional[str]
        Name of the colormap from cmocean package or from the standard matplotlib set.
        By default `Spectral_r` will be used.
    influence: float
        Radius of influence for interpolation, in meters.
    box: List[float]
        Map boundaries in -180 180 -90 90 format that will be used for interpolation (default [-180 180 -89 90]).
    res: List[int]
        Number of points along each axis that will be used for interpolation (for lon and lat),
        default [360, 180].
    interp: str
        Interpolation method. Options are 'nn' (nearest neighbor), 'idist' (inverse distance), "linear" and "cubic".
    mapproj: str
        Map projection. Options are Mercator (merc), Plate Carree (pc),
        North Polar Stereo (np), South Polar Stereo (sp),  Robinson (rob)
    levels: Optional[List]
        Levels for contour plot in format (min, max, numberOfLevels). List with more than
        3 values will be interpreted as just a list of individual level values.
        If not provided min/max values from data will be used with 40 levels.
    ptype: str
        Plot type. Options are contourf (\'cf\') and pcolormesh (\'pcm\')
    units: Optional[str]
        Units for color bar.
    figsize: Tuple[int, int]
        Figure size in inches
    rowscol: Tuple[int, int]
        Number of rows and columns.
    titles: Optional[Union[str, List[str]]]
        Title of the plot (if string) or subplots (if list of strings)
    distances_path: Optional[str]
        Path to the file with distances. If not provided and dumpfile=True, it will be created.
    inds_path: Optional[str]
        Path to the file with inds. If not provided and dumpfile=True, it will be created.
    qhull_path: Optional[str]
        Path to the file with qhull (needed for linear and cubic interpolations).
        If not provided and dumpfile=True, it will be created.
    interpolated_data: Optional[np.ndarray]
        Data interpolated to regular grid (you also have to provide lonreg and latreg).
        If provided, data will be plotted directly, without interpolation.
    lonreg: Optional[np.ndarray]
        1D array of longitudes. Used in combination with `interpolated_data`,
        when you need to plot interpolated data directly.
    latreg: Optional[np.ndarray]
        1D array of latitudes. Used in combination with `interpolated_data`,
        when you need to plot interpolated data directly.
    basepath: Optional[str]
        Path where to store additional interpolation files. If None (default),
        the path of the mesh will be used.
    no_pi_mask: bool
        Mask PI by default or not.

    Returns
    -------
    Union[plt.Axes, List[plt.Axes]]
        Matplotlib axes objects
    """
    if not CARTOPY_AVAILABLE:
        raise ImportError("Cartopy is required for plotting")
        
    # Ensure data is a list
    if not isinstance(data, list):
        data = [data]
        
    # Validate titles
    if titles:
        if not isinstance(titles, list):
            titles = [titles]
        if len(titles) != len(data):
            raise ValueError(
                "The number of titles do not match the number of data fields, please adjust titles (or put to None)"
            )

    # Check if we have enough subplots
    if (rowscol[0] * rowscol[1]) < len(data):
        raise ValueError(
            "Number of rows*columns is smaller than number of data fields, please adjust rowscol."
        )

    # Get colormap
    colormap = get_cmap(cmap=cmap)

    # Set radius of influence
    radius_of_influence = influence

    # Get box boundaries
    left, right, down, up = box
    lonNumber, latNumber = res

    # Create grid
    if lonreg is None:
        lonreg = np.linspace(left, right, lonNumber)
        latreg = np.linspace(down, up, latNumber)

    lonreg2, latreg2 = np.meshgrid(lonreg, latreg)

    # Interpolate data if needed
    if interpolated_data is None:
        interpolated = interpolate_for_plot(
            data,
            mesh,
            lonreg2,
            latreg2,
            interp=interp,
            distances_path=distances_path,
            inds_path=inds_path,
            radius_of_influence=radius_of_influence,
            basepath=basepath,
            qhull_path=qhull_path,
        )
    else:
        interpolated = [interpolated_data]

    # Apply mask
    m2 = mask_ne(lonreg2, latreg2)

    for i in range(len(interpolated)):
        if not no_pi_mask:
            interpolated[i] = np.ma.masked_where(m2, interpolated[i])
        interpolated[i] = np.ma.masked_equal(interpolated[i], 0)

    # Create figure and axes
    fig, ax = create_proj_figure(mapproj, rowscol, figsize)

    if isinstance(ax, np.ndarray):
        ax = ax.flatten()
    else:
        ax = [ax]

    # Plot each dataset
    for ind, data_int in enumerate(interpolated):
        ax[ind].set_extent([left, right, down, up], crs=ccrs.PlateCarree())

        data_levels = get_plot_levels(levels, data_int, lev_to_data=False)

        if ptype == "cf":
            # Add cyclic point to avoid discontinuities at the date line
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
            mmin = data_levels[0]
            mmax = data_levels[-1]
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
            raise ValueError(f"Unknown plot type '{ptype}'. Use 'cf' or 'pcm'.")

        # Add coastlines
        ax[ind].add_feature(
            cfeature.GSHHSFeature(levels=[1], scale="low", facecolor="lightgray")
        )
        
        # Add title if provided
        if titles:
            titles_copy = titles.copy()
            ax[ind].set_title(titles_copy.pop(0), size=20)

    # Remove unused subplots
    for delind in range(ind + 1, len(ax)):
        fig.delaxes(ax[delind])

    # Add colorbar
    cb = fig.colorbar(
        image, orientation="horizontal", ax=ax, pad=0.01, shrink=0.9, format=sfmt
    )

    cb.ax.tick_params(labelsize=15)

    if units:
        cb.set_label(units, size=20)

    return ax



def plot_vector(
    u_int: Union[np.ndarray, List[np.ndarray]],
    v_int: Union[np.ndarray, List[np.ndarray]],
    lonreg2: np.ndarray,
    latreg2: np.ndarray,
    box: List[float] = [-180, 180, -80, 90],
    cmap: Optional[str] = None,
    rowscol: List[int] = [1, 1],
    figsize: Tuple[int, int] = (15, 10),
    titles: Optional[Union[str, List[str]]] = None,
    sstep: int = 3,
    scale: int = 20,
    vmin: float = 0,
    vmax: float = 0.5,
    mapproj: str = "pc",
    regrid_shape: Optional[int] = None,
    units: str = "m/s",
) -> Union[plt.Axes, List[plt.Axes]]:
    """Plot vector field on the map.
    
    Parameters
    ----------
    u_int: Union[np.ndarray, List[np.ndarray]]
        U component of the vector field
    v_int: Union[np.ndarray, List[np.ndarray]]
        V component of the vector field
    lonreg2: np.ndarray
        2D array of longitudes
    latreg2: np.ndarray
        2D array of latitudes
    box: List[float]
        Map boundaries in -180 180 -90 90 format
    cmap: Optional[str]
        Name of the colormap
    rowscol: List[int]
        Number of rows and columns for subplots
    figsize: Tuple[int, int]
        Figure size in inches
    titles: Optional[Union[str, List[str]]]
        Title(s) for the plot(s)
    sstep: int
        Stride for decimating the vector field
    scale: int
        Scale for the arrows
    vmin: float
        Minimum value for color normalization
    vmax: float
        Maximum value for color normalization
    mapproj: str
        Map projection
    regrid_shape: Optional[int]
        Regrid shape parameter for quiver
    units: str
        Units for the colorbar
        
    Returns
    -------
    Union[plt.Axes, List[plt.Axes]]
        Matplotlib axes objects
    """
    if not CARTOPY_AVAILABLE:
        raise ImportError("Cartopy is required for plotting")
        
    # Ensure inputs are lists
    if not isinstance(u_int, list):
        u_int = [u_int]
    if not isinstance(v_int, list):
        v_int = [v_int]

    # Validate titles
    if titles:
        if not isinstance(titles, list):
            titles = [titles]
        if len(titles) != len(u_int):
            raise ValueError(
                "The number of titles do not match the number of data fields, please adjust titles (or put to None)"
            )

    # Check if we have enough plots
    if (rowscol[0] * rowscol[1]) < len(u_int):
        raise ValueError(
            "Number of rows*columns is smaller than number of data fields, please adjust rowscol."
        )
        
    # Get colormap and normalization
    colormap = get_cmap(cmap=cmap)
    normalisation = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    
    # Get box boundaries
    left, right, down, up = box
    
    # Create figure and axes
    fig, ax = create_proj_figure(mapproj, rowscol=rowscol, figsize=figsize)

    if isinstance(ax, np.ndarray):
        ax = ax.flatten()
    else:
        ax = [ax]

    # Plot each vector field
    for ind, (u_data, v_data) in enumerate(zip(u_int, v_int)):
        ax[ind].set_extent([left, right, down, up], crs=ccrs.PlateCarree())
        
        # Calculate speed for coloring
        speed = np.hypot(u_data, v_data)
        
        # Plot the vector field
        image = ax[ind].quiver(
            lonreg2[::sstep, ::sstep],
            latreg2[::sstep, ::sstep],
            u_data[::sstep, ::sstep],
            v_data[::sstep, ::sstep],
            speed[::sstep, ::sstep],
            transform=ccrs.PlateCarree(),
            scale=scale,
            norm=normalisation,
            regrid_shape=regrid_shape,
            cmap=colormap,
        )
        
        # Add coastlines
        ax[ind].coastlines(resolution="50m", lw=0.5)
        
        # Add title if provided
        if titles and ind < len(titles):
            ax[ind].set_title(titles[ind], size=20)
    
    # Add colorbar
    cb = fig.colorbar(
        image, orientation="horizontal", ax=ax, pad=0.01, shrink=0.9, format=sfmt
    )
    cb.ax.tick_params(labelsize=15)
    
    if units:
        cb.set_label(units, size=20)

    return ax



def plot_transect_map(
    lonlat: np.ndarray,
    mesh: Any,
    view: str = "w",
    stock_img: bool = False,
    size: float = 30
) -> plt.Axes:
    """Plot map of the transect.

    Parameters
    ----------
    lonlat: np.ndarray
        2 dimensional np.array that contains longitudes and latitudes.
        Can be constructed from vectors as lonlat = np.vstack((lon, lat))
    mesh: Any
        FESOM2 mesh object
    view: str
        Projection to use for the map:
        w - global (Mercator)
        np - North Polar Stereo
        sp - South Polar Stereo
    stock_img: bool
        Show stock background image. Usually makes things slower.
    size: float
        Size of the points on the map.

    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    if not CARTOPY_AVAILABLE:
        raise ImportError("Cartopy is required for plotting")
    
    # Get nodes for the transect
    nodes = transect_get_nodes(lonlat, mesh)

    # Create figure with appropriate projection
    if view == "w":
        ax = plt.subplot(111, projection=ccrs.Mercator(central_longitude=0))
        ax.set_global()
    elif view == "np":
        ax = plt.subplot(111, projection=ccrs.NorthPolarStereo(central_longitude=0))
        ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
    elif view == "sp":
        ax = plt.subplot(111, projection=ccrs.SouthPolarStereo(central_longitude=0))
        ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    else:
        raise ValueError(
            f'The "{view}" is not recognized as valid view option. Use "w", "np", or "sp".'
        )

    # Plot the transect points and the mesh nodes
    ax.scatter(lonlat[0, :], lonlat[1, :], s=size, c="b", transform=ccrs.PlateCarree())
    ax.scatter(
        mesh.x2[nodes], mesh.y2[nodes], s=size, c="r", transform=ccrs.PlateCarree()
    )
    
    # Add background and coastlines
    if stock_img:
        ax.stock_img()
    ax.coastlines(resolution="50m")
    
    return ax


def plot_transect(*args, **kwargs):
    """Deprecated function for plotting transects."""
    raise DeprecationWarning(
        "The plot_transect function is deprecated. Use combination of get_transect and plot_xyz instead."
    )


def xyz_plot_one(
    mesh: Any,
    data: np.ndarray,
    xvals: np.ndarray,
    levels: Optional[List] = None,
    maxdepth: int = 1000,
    label: str = r"$^{\circ}$C",
    title: str = "",
    cmap: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    facecolor: str = "lightgray",
    fontsize: int = 12,
    xlabel: str = "Time"
) -> plt.Artist:
    """Plot a single depth/x cross-section."""
    # Get depth index based on maxdepth
    depth_index = ind_for_depth(maxdepth, mesh)

    # Create or use axes
    if ax is None:
        ax = plt.gca()
        oneplot = True
    else:
        oneplot = False

    # Get colormap
    colormap = get_cmap(cmap=cmap)
    
    # Get levels for the plot - no need to transpose with new data shape
    data_levels = get_plot_levels(levels, data[:depth_index, :], lev_to_data=False)

    # Create the contour plot - no need to transpose with new data shape
    image = ax.contourf(
        xvals,
        np.abs(mesh.zlev[:depth_index]),
        data[:depth_index, :],  # Changed from data[:, :depth_index].T
        levels=data_levels,
        cmap=colormap,
        extend="both",
    )
    
    # Configure the plot
    ax.invert_yaxis()
    ax.set_title(title, size=fontsize)
    ax.set_xlabel(xlabel, size=fontsize)
    ax.set_ylabel("Depth, m", size=fontsize)
    ax.set_facecolor(facecolor)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)

    # Add colorbar if this is a standalone plot
    if oneplot:
        cb = plt.colorbar(image, format=sfmt)
        cb.set_label(label, size=fontsize)
        cb.ax.tick_params(labelsize=fontsize)
        cb.ax.yaxis.get_offset_text().set_fontsize(fontsize)

    return image


def xyz_plot_many(
    mesh: Any,
    data: List[np.ndarray],
    xvals: np.ndarray,
    levels: Optional[List] = None,
    maxdepth: int = 1000,
    label: str = r"$^{\circ}$C",
    title: Union[str, List[str]] = "",
    cmap: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    facecolor: str = "lightgray",
    fontsize: int = 12,
    ncols: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    xlabel: str = "Time"
) -> plt.Figure:
    """Plot multiple depth/x cross-sections.
    
    Parameters
    ----------
    mesh: Any
        FESOM2 mesh object
    data: List[np.ndarray]
        List of 2D data arrays to plot (depth, x)  # Updated shape description
    xvals: np.ndarray
        Values for x-axis (e.g., time or distance)
    levels: Optional[List]
        Level specifications for contour plot
    maxdepth: int
        Maximum depth for the plot
    label: str
        Label for the colorbar
    title: Union[str, List[str]]
        Title for the plots. If list, one title per plot
    cmap: Optional[str]
        Colormap name
    ax: Optional[plt.Axes]
        Ignored. Kept for API compatibility
    facecolor: str
        Background color for plots
    fontsize: int
        Font size for text elements
    ncols: int
        Number of columns in the figure
    figsize: Optional[Tuple[float, float]]
        Figure size (width, height) in inches
    xlabel: str
        Label for x-axis
        
    Returns
    -------
    plt.Figure
        The figure object
    """
    # Get depth index based on maxdepth
    depth_index = ind_for_depth(maxdepth, mesh)

    # Calculate number of rows and columns
    ncols = float(ncols)
    nplots = len(data)
    nrows = math.ceil(nplots / ncols)
    ncols = int(ncols)
    nrows = int(nrows)

    # Create figure with appropriate size
    if not figsize:
        figsize = (8 * ncols, 2 * nrows * ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    ax = ax.flatten()

    # Get colormap
    colormap = get_cmap(cmap=cmap)

    # Create plots for each dataset
    for ind, data_one in enumerate(data):
        # Get levels for this plot - no need to transpose
        data_levels = get_plot_levels(
            levels, data_one[:depth_index, :], lev_to_data=False
        )

        # Create the contour plot - no need to transpose
        image = ax[ind].contourf(
            xvals,
            np.abs(mesh.zlev[:depth_index]),
            data_one[:depth_index, :],
            levels=data_levels,
            cmap=colormap,
            extend="both",
        )
        
        # Configure the plot
        ax[ind].invert_yaxis()

        # Set title
        if not isinstance(title, list):
            ax[ind].set_title(title, size=fontsize)
        else:
            ax[ind].set_title(title[ind], size=fontsize)

        # Set labels and style
        ax[ind].set_xlabel(xlabel, size=fontsize)
        ax[ind].set_ylabel("Depth, m", size=fontsize)
        ax[ind].set_facecolor(facecolor)
        ax[ind].tick_params(axis="both", which="major", labelsize=fontsize)

        # Add colorbar for each plot
        cb = fig.colorbar(
            image, orientation="horizontal", ax=ax[ind], pad=0.11, format=sfmt
        )
        cb.set_label(label, size=fontsize)
        cb.ax.tick_params(labelsize=fontsize)
        cb.ax.xaxis.get_offset_text().set_fontsize(fontsize)

    # Remove unused subplots
    for delind in range(ind + 1, len(ax)):
        fig.delaxes(ax[delind])

    # Adjust layout
    fig.tight_layout()

    return fig


def hofm_plot(*args, **kwargs):
    """Deprecated function for Hovmöller plots."""
    raise DeprecationWarning(
        "The hovm_plot function is deprecated. Use plot_xyz instead."
    )


def plot_xyz(
    mesh: Any,
    data: Union[np.ndarray, xr.DataArray, List[Union[np.ndarray, xr.DataArray]]],
    xvals: Optional[np.ndarray] = None,
    levels: Optional[List] = None,
    maxdepth: int = 1000,
    label: str = r"$^{\circ}$C",
    title: Union[str, List[str]] = "",
    cmap: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    facecolor: str = "lightgray",
    fontsize: int = 12,
    ncols: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    xlabel: str = "Time",
) -> Union[plt.Artist, plt.Figure]:
    """Plot data on x (e.g. time, distance) / depth.

    Parameters
    ----------
    mesh: Any
        pyfesom2 mesh object
    data: Union[np.ndarray, xr.DataArray, List[Union[np.ndarray, xr.DataArray]]]
        2D input data. Can be either one or several (as a list)
        numpy arrays or xarray DataArrays. If list of arrays is
        provided, several plots will be plotted at once in a multipanel.
    xvals: Optional[np.ndarray]
        Values for the x axis (e.g. time, distance).
        Deduced automatically if `data` is xarray DataArray.
        Should be provided if `data` is nd array.
    levels: Optional[List]
        Levels for contour plot in format (min, max, numberOfLevels). List with more than
        3 values will be interpreted as just a list of individual level values.
        If not provided min/max values from data will be used with 40 levels.
    maxdepth: int
        Maximum depth the plot will be limited to
    label: str
        Label for colorbar
    title: Union[str, List[str]]
        Should be str if only one plot is expected.
        For multipanel plots should be the list of strings.
    cmap: Optional[str]
        Matplotlib colormap instance
    ax: Optional[plt.Axes]
        Only for single plot. It can be inserted to other figure.
    facecolor: str
        Used to fill areas with NaNs.
        Should be the name of the color that matplotlib can understand.
    fontsize: int
        Font size of text elements (e.g. labels)
    ncols: int
        Number of columns for multipanel plot.
    figsize: Optional[Tuple[float, float]]
        ONLY works for multipanel plots.
        For single plots use plt.figure(figsize=(10, 10))
        before calling this function.
    xlabel: str
        Label for x axis.
        
    Returns
    -------
    Union[plt.Artist, plt.Figure]
        For single plot returns the contourf artist, for multiple plots returns the figure
    """
    # Handle single dataset case
    if not isinstance(data, list):
        # Extract xvals from xarray if needed
        if isinstance(data, xr.DataArray):
            xvals = data.time.data
        else:
            if xvals is None:
                raise ValueError(
                    "You provide np.array as an input, but did not provide xvals (e.g. time or distance)"
                )

        # Plot single dataset
        return xyz_plot_one(
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
    # Handle multiple datasets case
    else:
        # Extract xvals from xarray if needed
        if isinstance(data[0], xr.DataArray):
            xvals = data[0].time.data
        else:
            if xvals is None:
                raise ValueError(
                    "You provide np.array as an input, but did not provide xvals (e.g. time or distance)"
                )

        # Plot multiple datasets
        return xyz_plot_many(
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
    mesh: Any,
    data: Union[np.ndarray, List[np.ndarray]],
    cmap: Optional[str] = None,
    box: List[float] = [-180, 180, -80, 90],
    mapproj: str = "pc",
    levels: Optional[List] = None,
    ptype: str = "cf",
    units: str = r"$^\circ$C",
    figsize: Tuple[int, int] = (10, 10),
    rowscol: Tuple[int, int] = (1, 1),
    titles: Optional[Union[str, List[str]]] = None,
    lw: float = 0.01,
    fontsize: float = 12,
    box_expand: float = 1,
) -> Union[plt.Axes, List[plt.Axes]]:
    """Plots original field on the cartopy map using tricontourf or tripcolor.

    Parameters
    ----------
    mesh: Any
        FESOM2 mesh object
    data: Union[np.ndarray, List[np.ndarray]]
        FESOM 2 data on nodes
        (for u,v,u_ice and v_ice one has to first interpolate
        from elements to nodes (`tonodes` function)).
        Can be either one np.ndarray or list of np.ndarrays.
    cmap: Optional[str]
        Name of the colormap from cmocean package or from the
        standard matplotlib set.
        By default `Spectral_r` will be used.
    box: List[float]
        Map boundaries in -180 180 -90 90 format that will be used for data
        selection and plotting (default [-180 180 -89 90]).
    mapproj: str
        Map projection. Options are Mercator (merc), Plate Carree (pc),
        North Polar Stereo (np), South Polar Stereo (sp),  Robinson (rob)
    levels: Optional[List]
        Levels for contour plot in format (min, max, numberOfLevels). List with more than
        3 values will be interpreted as just a list of individual level values.
        If not provided min/max values from data will be used with 40 levels.
    ptype: str
        Plot type. Options are tricontourf (\'cf\') and tripcolor (\'tri\')
    units: str
        Units for color bar.
    figsize: Tuple[int, int]
        Figure size in inches
    rowscol: Tuple[int, int]
        Number of rows and columns.
    titles: Optional[Union[str, List[str]]]
        Title of the plot (if string) or subplots (if list of strings)
    lw: float
        Line width for tripcolor plot edges
    fontsize: float
        Font size of some of the plot elements.
    box_expand: float
        How much bigger the selected part of the mesh should be
        compared to the `box` to avoid white boundaries.
        Value is in degrees and default is 1.
        
    Returns
    -------
    Union[plt.Axes, List[plt.Axes]]
        Matplotlib axes objects
    """
    if not CARTOPY_AVAILABLE:
        raise ImportError("Cartopy is required for plotting")
        
    # Ensure data is a list
    if not isinstance(data, list):
        data = [data]
        
    # Validate titles
    if titles:
        if not isinstance(titles, list):
            titles = [titles]
        if len(titles) != len(data):
            raise ValueError(
                "The number of titles do not match the number of data fields, please adjust titles (or put to None)"
            )

    # Check if we have enough subplots
    if (rowscol[0] * rowscol[1]) < len(data):
        raise ValueError(
            "Number of rows*columns is smaller than number of data fields, please adjust rowscol."
        )

    # Get colormap
    colormap = get_cmap(cmap=cmap)
    
    # Expand box for mesh selection to avoid white boundaries
    box_mesh = [box[0] - box_expand, box[1] + box_expand, 
                box[2] - box_expand, box[3] + box_expand]

    # Create figure and axes
    fig, ax = create_proj_figure(mapproj, rowscol, figsize)
    if isinstance(ax, np.ndarray):
        ax = ax.flatten()
    else:
        ax = [ax]

    # Plot each dataset
    for ind, data_to_plot in enumerate(data):
        # Get levels for this plot
        data_levels = get_plot_levels(levels, data_to_plot, lev_to_data=True)
        
        # Set map extent
        ax[ind].set_extent(box, crs=ccrs.PlateCarree())
        
        # Cut mesh region and prepare for plotting
        elem_no_nan, no_nan_triangles = cut_region(mesh, box_mesh)
        no_cyclic_elem2 = get_no_cyclic(mesh, elem_no_nan)
        
        # Prepare data for plotting
        if data_to_plot.shape[0] == mesh.n2d:
            # Data on nodes
            data_to_plot = data_to_plot.copy()  # Avoid modifying original data
            data_to_plot[data_to_plot == 0] = -99999  # masked values don't work in cartopy
            elem_to_plot = elem_no_nan[no_cyclic_elem2]
        elif data_to_plot.shape[0] == mesh.e2d:
            # Data on elements
            if ptype == "cf":
                raise ValueError(
                    "You are trying to plot data on elements using countourf, this will not work. Use `ptype='tri'` instead."
                )
            data_to_plot = data_to_plot[no_nan_triangles][no_cyclic_elem2]
            data_to_plot = data_to_plot.copy()  # Avoid modifying original data
            data_to_plot[data_to_plot == 0] = np.nan
            elem_to_plot = elem_no_nan[no_cyclic_elem2]
        else:
            raise ValueError(
                f"Data size ({data_to_plot.shape[0]}) doesn't match either nodes ({mesh.n2d}) or elements ({mesh.e2d})"
            )

        # Create the plot
        if ptype == "tri":
            # Tripcolor plot
            image = ax[ind].tripcolor(
                mesh.x2,
                mesh.y2,
                elem_to_plot,
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
            # Tricontourf plot
            image = ax[ind].tricontourf(
                mesh.x2,
                mesh.y2,
                elem_to_plot,
                data_to_plot,
                levels=data_levels,
                transform=ccrs.PlateCarree(),
                cmap=colormap,
            )
        else:
            raise ValueError(
                "Only `cf` (contourf) and `tri` (tripcolor) options are supported."
            )

        # Add coastlines
        ax[ind].coastlines(lw=1.5, resolution="110m")

        # Add title if provided
        if titles:
            titles_copy = titles.copy()
            ax[ind].set_title(titles_copy.pop(0), size=20)

    # Remove unused subplots
    for delind in range(ind + 1, len(ax)):
        fig.delaxes(ax[delind])

    # Add colorbar
    cb = fig.colorbar(
        image, orientation="horizontal", ax=ax, pad=0.01, shrink=0.9, format=sfmt
    )

    cb.ax.tick_params(labelsize=fontsize)

    if units:
        cb.set_label(units, size=fontsize)

    return ax