"""
Pyfesom2 Xarray Accessors

This module provides additional functionality to FESOM2 xarray dataset. This additional functionality adds
selection, plotting, regridding for unstructured FESOM2 grid. These features are implemted using neat xarray accessor
protocol.

Examples:
    from pyfesom2 import open_dataset
    fesom_xr_dataset = open_dataset(data_path, mesh_path, ...)
    fesom_xr_dataset.pyfesom2.sel(...)
    fesom_xr_dataset.pyfesom2.plot(...)

Design Considerations:
Goal is to provide commonly used xarray methods (like sel, plot... ) that are currently un-supported for unstructured
FESOM model data. We would like to have these methods available on xarray dataset level and dataarray (data variable).

Implementing this would mean that we need to register a "pyfesom2" dataset and a dataarray accessor
(http://xarray.pydata.org/en/stable/api.html#advanced-api). Accessor on dataset is for convinince to do operations like
selection all variables and this is also efficient (more on that below), accessor on dataarray is most instutuve and
most apt for plotting: for example dataset.variable.pyfesom2.plot(..).

Having both dataset and datarray working on same kind of data has issues:
 1.) It creates a not-so-desirable user interface for certain operations, for instance, to select a region on dataset and
plot a variable: dataset.pyfesom2.sel(region.).variable.pyfesom2.plot(...).
 2.) Most user intuitive interface to plotting is via: ...variable.plot(...). For variables on this unstructured grid,
 generating spatial plots would need node connectivity information without which land and oceans cannot be shown
accurately and this additional information (like faces(3, nelems)) cannot be part of coordinates of dataarray of
variable. Xarray's dataarray object can only hold  coordinates for dims of variable(while a dataset can hold any arbitary
 coordinate information ). This would user interface to spatial plotting of variable should get mesh information
 externally, either as argument in plot method or path to mesh in dataarray's attribute. That is inefficient among
 other UI issues.
 3.) Selection on this unstructured grid is implemented using a tree, building a tree can be expensive depending on
 grid size. When using selection on a variable algorithm would build a tree and we would like to utilize that tree in
 selections on other variables in future. And to retain mesh info after a selection (see 2.) we should return a dataset,
 this is in contrast to xarray's default selection on a variable returning a dataarray.

Following implementation takes these and user interface into consideration:
pyfesom2 is registered as xarray dataset accessor only, access to data variables from this accessor are controlled
through a wrapping class to data variable (instead of dataarray accessor) that contains context of
dataset (thereby node info and is able to share selection tree). Logic of wrapping data variable in a class is what
accessors are behind the scenes in xarrray, except in this case we pass dataset context to the constructor. Need to
 watchout for thread safe parallel modifications to Dataset attribures.


example access patterns would be:
dataset.pyfesom2.sel(...) , dataset.pyfesom2.variable.sel(...)
dataset.pyfesom2.variable.triplot()

One other way to think for future would be to subclass xarray's Dataset and DataArray against their recommendation
but might be worth it in this case. It might need painful re-implement/tweak xarray's open_dataset kind of methods.
Another option could be to have a datastore for fesom data (not well-thought.

TODO: possibly add projection as a setting flag.
      set plot type interactive or not as setting.
"""

import functools
from collections.abc import Sequence  # python>3.3?
from typing import Union, Sequence as SequenceType, Tuple

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.tri import Triangulation
from shapely.geometry import MultiPolygon, Polygon, asMultiPoint

# New Types
BoundingBox = SequenceType[float]
MultiRegion = Union[SequenceType[Polygon], MultiPolygon]


# Selection functions

def distance_along_trajectory(lons, lats):
    from cartopy.geodesic import Geodesic
    geod = Geodesic()
    lons, lats = np.array(lons, ndmin=1, copy=False), np.array(lons, ndmin=1, copy=False)
    points = np.c_[lons, lats]
    dists = np.zeros(lons.shape[0])
    dists[1:] = np.cumsum(geod.inverse(points[0:-1], points[1:])[:, 0])
    return dists


def normalize_distance(distance_array_in_m):
    """Returns best representation for
    distances in m or km based on all values
    of array
    """
    distance_array_in_km = distance_array_in_m / 1000.0
    len_array = distance_array_in_m.shape[0]
    # if more then 1/3 of points are best suited to be expressed in m then m else km
    if np.count_nonzero(distance_array_in_km < 1) > len_array / 3:
        return "m", distance_array_in_m
    else:
        return "km", distance_array_in_km


def select_points(xrobj: Union[xr.Dataset, xr.DataArray],
                  lon, lat, method='nearest', tolerance=None, tree=None, return_distance=True, **other_dims) -> Union[
    xr.Dataset, xr.DataArray]:
    """

    Parameters
    ----------
    xrobj
    lon
    lat
    method
    tolerance
    tree

    Returns
    -------
    TODO: check id all dims are of same length.
    """
    from cartopy.crs import Geocentric, Geodetic
    from scipy.spatial import cKDTree
    if not method == 'nearest':
        raise NotImplementedError("Spatial selection currently supports only nearest neighbor lookup")
    geocentric_crs, geodetic_crs = Geocentric(), Geodetic()
    if tree is None:
        src_pts = geocentric_crs.transform_points(geodetic_crs, np.asarray(xrobj.lon), np.asarray(xrobj.lat))
        tree = cKDTree(src_pts, leafsize=32, compact_nodes=False, balanced_tree=False)

    dst_pts = geocentric_crs.transform_points(geodetic_crs, np.asarray(lon), np.asarray(lat))

    if tolerance is None:
        _, ind = tree.query(dst_pts)
    else:
        # inds = tree.query_ball_point(dst_pts, r=tolerance, n_jobs=-1)[0]
        # _, ind = tree.query(dst_pts, r=tolerance)
        # ind = inds[0]
        raise NotImplementedError('tolerance is currently not supported.')

    other_dims = {k: xr.DataArray(np.array(v, ndmin=1), dims='points') for k, v in other_dims.items()}
    retobj = xrobj.isel(nod2=xr.DataArray(ind-1, dims='points')).sel(**other_dims, method=method)

    if return_distance:
        dist = distance_along_trajectory(lon, lat)
        dist_units, dist = normalize_distance(dist)
        retobj = retobj.assign_coords({'distance': ('points', dist)})
        retobj.distance.attrs['units'] = dist_units
        retobj.distance.attrs['long_name'] = f"distance along trajectoru"
    return retobj


class SimpleMesh:
    """Wrapper that fakes pyfesom's mesh object for purposes of this module"""

    def __init__(self, lon, lat, faces):
        self.x2 = lon
        self.y2 = lat
        self.elem = faces


def select_bbox(ds: xr.Dataset, bbox: BoundingBox) -> xr.Dataset:
    """bbox as xmin,xmax, ymin, ymax"""
    from .ut import cut_region
    data = ds.drop('faces')
    mesh = SimpleMesh(ds.lon, ds.lat, ds.faces)
    cut_faces = cut_region(mesh, bbox)
    uniq, inv_index = np.unique(cut_faces.faces.values.ravel(), return_inverse=True)
    new_faces = inv_index.reshape(cut_faces.shape)
    ret = data.isel(nod2=uniq)
    ret = ret.assign_coords({'faces': (('nelem', 'three'), new_faces)})
    return ret


def select_region(xrobj: xr.Dataset,
                  region: Union[BoundingBox, Polygon]) -> xr.Dataset:
    from shapely.geometry import box, Polygon
    if isinstance(region, Sequence) and len(region) == 4:
        sel_polygon = box(*region)
    elif isinstance(region, Polygon):
        sel_polygon = region
    else:
        raise NotImplementedError(f'Supplied region data {region} is not yet supported')
    left, down, right, top = sel_polygon.bounds
    sel = select_bbox(xrobj, [left, right, down, top])
    if sel_polygon == sel_polygon.envelope:
        return sel

    as_mp = asMultiPoint(np.vstack((sel.lon, sel.lat)).T)  # asMultiPoint saves memory according to docs

    vals = np.asarray(as_mp.intersection(sel_polygon))

    in_lon, in_lat = vals.T
    sel_inds = (
        np.isin(sel.lon.values, in_lon, assume_unique=True) & np.isin(sel.lat.values, in_lat, assume_unique=True))

    mesh = SimpleMesh(sel.lon, sel.lat, sel.faces)
    nod2_inds = np.nonzero(sel_inds)  # nonzero works only for bool arrays to get inds
    # indices of nod2 that correspond to polygon
    new_elem = mesh.elem[np.mean(np.isin(mesh.elem, nod2_inds), axis=1) == 1]
    uniq, inv_index = np.unique(new_elem.values.ravel(), return_inverse=True)
    new_faces = inv_index.reshape(new_elem.shape)
    ret = sel.isel(nod2=uniq)
    ret = ret.assign_coords({'faces': (('nelem', 'three'), new_faces)})
    return ret


def sel_nn_points(ds, lon, lat):
    lon, lat = np.asarray(lon), np.asarray(lat)
    bounds = (lon.min(), lon.max(), lat.min(), lat.max())
    sel = select_bbox(ds, bounds)
    tris = Triangulation(sel.lon, sel.lat, triangles=sel.faces)
    tf = tris.get_trifinder()
    inds = tf(lon, lat)
    new_faces = sel.faces[inds[inds > -1]]
    inds, inv_index = np.unique(new_faces.values.ravel(), return_inverse=True)
    new_faces = inv_index.reshape(new_faces.shape)
    ret = sel.isel(nod2=inds)
    ret = ret.drop('faces')
    ret = ret.assign_coords({'faces': (('nelem', 'three'), new_faces)})
    return ret


def select(xrobj: Union[xr.Dataset, xr.DataArray], method='nearest',
           tolerance=None, region=None, path=None, **indexers) -> Union[xr.Dataset, xr.DataArray]:
    """
    Wrapper that does selection
    Parameters
    ----------
    xrobj
    method
    tolerance
    region
    path
    indexers
    Returns
    -------

    """
    lat = indexers.pop('lat', None)
    lon = indexers.pop('lon', None)
    lat_indexer = True if lat is not None else False
    lon_indexer = True if lon is not None else False

    if (lat_indexer or lon_indexer) and (region is not None or path is not None):
        # TODO: do this combinations better, doesn't check if path and region are both given
        raise Exception("Onlu one option: lat, lon as indexer or path or region is supported")

    if lat_indexer or lon_indexer:
        if lat_indexer and lon_indexer:
            if method == 'nearest':
                ret_arr = select_points(xrobj, lon, lat, method=method, tolerance=tolerance)
                ret_arr = ret_arr.drop_vars('faces')
            else:
                raise NotImplementedError("Only method='nearest' is currently supported.")
        else:
            raise NotImplementedError("Both lat, lon are needed as indexers, else use path or region.")
    elif region is not None:
        ret_arr = select_region(xrobj, region)
    elif path is not None:
        raise NotImplementedError('Path option is not implemented yet')
    else:
        ret_arr = xrobj

    return ret_arr.sel(**indexers, method=method).squeeze()


# Accessors

@xr.register_dataset_accessor("pyfesom2")
class FESOMDataset:
    def __init__(self, xr_dataset: xr.Dataset):
        self._xrobj = xr_obj = xr_dataset
        # TODO: check valid fesom data?
        self._tree = None
        for datavar in xr_obj.data_vars:
            # setattr(self, str(datavar), xr_obj[datavar])
            setattr(self, str(datavar), FESOMDataArray(xr_obj[datavar], xr_obj))

    def select(self, method='nearest', tolerance=None, region=None, path=None, **indexers):
        sel_obj = select(self._xrobj, method='nearest', tolerance=tolerance, region=region, path=path, **indexers)
        return sel_obj

    def regrid(self):
        pass

    def regrid_like(self):
        pass

    def plot(self, *args, **kwargs):
        return self._xrobj.plot(*args, **kwargs)

    def triplot(self, *args, **kwargs):
        """
        Plot mesh
        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        data = self._xrobj
        tri = Triangulation(data.lon, data.lat, triangles=data.faces)
        projection = kwargs.pop('projection', ccrs.PlateCarree())
        ax = kwargs.pop('ax', plt.axes(projection=projection))
        return ax.triplot(tri, *args, **kwargs)

    def __repr__(self):
        return self._xrobj.__repr__()

    def _repr_html_(self):
        return self._xrobj._repr_html_()


def trimesh_plot(darr, tris, **other_dims):
    import geoviews as gv
    tris = np.asarray(tris)
    data = darr.sel(**other_dims)
    var_name = data.name
    var_da = gv.Dataset((darr.lon, darr.lat, data),
                        kdims=['lon', 'lat'], vdims=[var_name])
    return gv.TriMesh((tris, var_da))


class FESOMDataArray:
    """ A Awapper around Dataarray"""

    def __init__(self, xr_dataarray: xr.DataArray, context_dataset=None):
        self._xrobj = xrobj = xr_dataarray
        self._context_dataset = context = context_dataset
        self._native_projection = ccrs.PlateCarree()

    def __repr__(self):
        return f"Wrapped {self._xrobj.__repr__()}\n{super().__repr__()}"

    def _repr_html_(self):
        return self._xrobj._repr_html_()

    def tripcolor(self, *args, **kwargs):
        data = self._xrobj.squeeze()
        if len(data.dims) > 1 or "nod2" not in data.dims:
            raise Exception('Not a spatial dataset')

        projection = kwargs.pop('projection', None)
        projection, tri = self._triangulate(data, projection)

        ax = kwargs.pop('ax', plt.axes(projection=projection))

        minv, maxv = data.min().values, data.max().values

        data = data.fillna(minv - 9999)  # make sure it is out of data bounds
        colorbar = kwargs.pop('colorbar', True)
        pl = ax.tripcolor(tri, data, *args, **kwargs)

        if colorbar:
            plt.colorbar(pl, ax=ax)
        return pl

    def tricontourf(self, *args, **kwargs):
        data = self._xrobj.squeeze()
        if len(data.dims) > 1 or "nod2" not in data.dims:
            raise Exception('Not a spatial dataset')

        projection = kwargs.pop('projection', None)

        projection, tri = self._triangulate(data, projection)

        ax = kwargs.pop('ax', plt.axes(projection=projection))

        if 'extents' in kwargs:
            ax.set_extent(kwargs.pop('extents'), crs=ccrs.PlateCarree())

        if 'extents' in kwargs:
            ax.set_extent(kwargs.pop('extents'), crs=ccrs.PlateCarree())
            ax.set_global()

        minv, maxv = data.min().values, data.max().values

        # pl=ax.tricontourf(tri, data, levels=np.linspace(minv, maxv,100))
        data = data.fillna(minv - 9999)  # make sure missing vals are out of data bounds

        levels = kwargs.pop('levels', np.linspace(minv, maxv, 100))
        colorbar = kwargs.pop('colorbar', True)

        pl = ax.tricontourf(tri, data, *args, levels=levels, **kwargs)

        if colorbar:
            plt.colorbar(pl, ax=ax)
        return pl

    def _triangulate(self, data, projection) -> Tuple[ccrs.Projection, Triangulation]:
        if projection:
            # transform_points cannot take DataArray unlike Triangulation
            tr_x, tr_y, _ = projection.transform_points(self._native_projection, data.lon.values, data.lat.values).T
            tri = Triangulation(tr_x, tr_y, triangles=self._context_dataset.faces)
        else:  # grid native projection
            projection = ccrs.PlateCarree()
            tri = Triangulation(data.lon, data.lat, triangles=self._context_dataset.faces)
        return projection, tri

    def tricontour(self, *args, **kwargs):
        data = self._xrobj.squeeze()
        if len(data.dims) > 1 or "nod2" not in data.dims:
            raise Exception('Not a spatial dataset')

        projection = kwargs.pop('projection', None)
        projection, tri = self._triangulate(data, projection)

        ax = kwargs.pop('ax', plt.axes(projection=projection))

        if 'extents' in kwargs:
            ax.set_extent(kwargs.pop('extents'), crs=ccrs.PlateCarree())

        minv, maxv = data.min().values, data.max().values

        # pl=ax.tricontourf(tri, data, levels=np.linspace(minv, maxv,100))
        data = data.fillna(minv - 9999)  # make sure it is out of data bounds
        levels = kwargs.pop('levels', np.linspace(minv, maxv, 100))
        colorbar = kwargs.pop('colorbar', True)
        pl = ax.tricontour(tri, data, *args, levels=levels, *args, **kwargs)

        if colorbar:
            plt.colorbar(pl, ax=ax)
        return pl

    def triplot(self, *args, **kwargs):
        data = self._xrobj

        projection = kwargs.pop('projection', None)
        projection, tri = self._triangulate(data, projection)

        ax = kwargs.pop('ax', plt.axes(projection=projection))

        if 'extents' in kwargs:
            ax.set_extent(kwargs.pop('extents'), crs=ccrs.PlateCarree())

        return ax.triplot(tri, *args, **kwargs)

    def trimesh(self, levels=None, cmap='RdBu', colorbar=True, height=350, width=600,
                colorbar_position="bottom", projection=None, coastline=True, tools=['hover'], **hv_kwopts):
        try:
            import geoviews as gv
            import holoviews as hv
            from holoviews.operation.datashader import rasterize
            hv.extension('bokeh')
        except ImportError as ex:
            raise ImportError('Using trimesh needs geoviews[holoviews, bokeh] and datashader') from ex

        trimesh_fn = functools.partial(trimesh_plot, darr=self._xrobj, tris=self._context_dataset.faces)
        var_name = self._xrobj.name
        hvd = hv.Dataset(self._xrobj.drop_vars(['lon', 'lat']))
        non_plot_dims = {dim:self._xrobj[dim].values for dim in self._xrobj.dims if dim!='nod2'}
        dmap = hv.DynamicMap(trimesh_fn,
                             kdims=list(non_plot_dims.keys())).redim.values(**non_plot_dims)
        if 'nz1' in non_plot_dims.keys():
             dmap = dmap.redim.default(nz1=self._xrobj.nz1.values[0])
             dmap = dmap.redim.unit(nz1='m')
             dmap = dmap.redim.label(nz1='level')

        plot_opts = {**hv_kwopts}
        if projection:
            plot_opts.update({'projection': projection})
        else:
            plot_opts.update({'projection': ccrs.PlateCarree()})
        if levels:
            if isinstance(levels, int):
                plot_opts.update({'color_levels': levels})
            elif isinstance(levels, Sequence):
                if len(levels) > 2:
                    plot_opts.update({'color_levels': len(levels)})
                    dmap = dmap.redim.range(var_name=(min(levels), max(levels)))
                elif len(levels) == 2:
                    dmap = dmap.redim.range(var_name=levels)
            else:
                raise Exception('Invalid levels, should be a tuple with limits or a sequence of values')
        else:
            dmap = dmap.opts(framewise=True)

        plot = rasterize(dmap).opts(width=width,
                                    height=height,
                                    cmap=cmap, colorbar=colorbar, tools=tools, **plot_opts)
        if coastline:
            return plot * gv.feature.coastline
        return plot

    def plot_transect(self, lon, lat, plot_type='auto', **indexers_plot_kwargs):
        """
        default plot_type
        """
        default_plot_types = ('line', 'contourf')  # 1d and 2d defualts
        extra_dims = []
        extra_dims = [k for k in indexers_plot_kwargs.keys() if
                      k in self._xrobj.coords]  # coords is used insead of dims because xarray will raise error in selection
        extra_indexers = {dim: indexers_plot_kwargs.pop(dim) for dim in extra_dims}
        sel = select_points(self._xrobj, lon=lon, lat=lat, **extra_indexers)
        dim_len = len(sel.dims)  # determines plot type, 1d or 2d

        # make time as default bottom x-axis if present (else formatting gets hard)
        xdims = ('time', 'distance') if 'time' in extra_dims else ('distance', None)
        if 'time' in extra_dims:
            sel['points'] = sel.time
        else:
            sel['points'] = sel.distance

        sel = sel.transpose(..., 'points')  # push points to last for making it default xaxis for plots

        # use xarray plotting as it formats datetime axis with ease and defaults cbar...
        if dim_len == 1:
            plot_type = default_plot_types[0] if plot_type == 'auto' else plot_type
            plot_fn = getattr(sel.plot, plot_type)
            plot = plot_fn(**indexers_plot_kwargs)[0]
        elif dim_len == 2:
            plot_type = default_plot_types[1] if plot_type == 'auto' else plot_type
            plot_fn = getattr(sel.plot, plot_type)
            plot = plot_fn(**indexers_plot_kwargs)
        else:
            raise Exception('A 2-d plot can have 1-2 dimensions, variable {sel.name} had {sel.dims}.')

        ax = plot.ax if hasattr(plot, 'ax') else plot.axes  # akward some plots have it as ax, some as axes

        if xdims[0] == 'time':
            ax2 = ax.twiny()
            ax2.set_xlim(sel.distance.min(), sel.distance.max())
            ax2.set_xlabel(f'distance [{sel.distance.units}]')
        return plot

    def select(self, method='nearest', tolerance=None, region=None, path=None, **indexers):
        sel_obj = self._xrobj.to_dataset()
        sel_obj = sel_obj.assign_coords({'faces': (self._context_dataset.faces.dims,
                                                   self._context_dataset.faces.values)})
        sel_obj = select(sel_obj, method='nearest', tolerance=tolerance, region=region, path=path, **indexers)
        return sel_obj
