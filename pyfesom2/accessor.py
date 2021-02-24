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

TODO:
      set plot type interactive or not as setting.
"""

import functools
import warnings
from typing import Optional, Sequence, Tuple, Union, MutableMapping

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.tri import Triangulation
from shapely.geometry import MultiPolygon, Polygon, LineString

# New Types
BoundingBox = Sequence[float]
Region = Union[BoundingBox, Polygon]
MultiRegion = Union[Sequence[Polygon], MultiPolygon]


# Selection

## Utilities for selection

def distance_along_trajectory(lons, lats):
    from cartopy.geodesic import Geodesic
    geod = Geodesic()
    lons, lats = np.array(lons, ndmin=1, copy=False), np.array(lats, ndmin=1, copy=False)
    points = np.c_[lons, lats]
    dists = np.zeros(lons.shape[0])
    temp_dist = geod.inverse(points[0:-1], points[1:])[:, 0]
    dists[1:] = np.cumsum(temp_dist)
    return dists


def normalize_distance(distance_array_in_m):
    """Returns best representation for
    distances in m or km.
    """
    distance_array_in_km = distance_array_in_m / 1000.0
    len_array = distance_array_in_m.shape[0]
    # if more then 1/3 of points are best suited to be expressed in m else in km
    if np.count_nonzero(distance_array_in_km < 1) > len_array // 3:
        return "m", distance_array_in_m
    else:
        return "km", distance_array_in_km


class SimpleMesh:
    """Wrapper that fakes pyfesom's mesh object for purposes of this module"""

    def __init__(self, lon, lat, faces):
        self.x2 = lon
        self.y2 = lat
        self.elem = faces


## Selection functions

def select_bbox(xr_obj: Union[xr.DataArray, xr.Dataset],
                bbox: BoundingBox,
                faces: Optional[Union[np.ndarray, xr.DataArray]] = None) -> xr.Dataset:
    """bbox as xmin, ymin, xmax, ymax
    doesn't tke shapely as input"""
    from .ut import cut_region
    faces = getattr(xr_obj, "faces", faces)
    if faces is None:
        raise ValueError(f"When passing a dataset it needs have faces in coords, or"
                         f"faces need to be passed explicitly.\n"
                         f"When passing a data array, argument faces can't be None,"
                         f"faces must be indices[nelem,3] that define triangles.")

    mesh = SimpleMesh(xr_obj.lon, xr_obj.lat, faces)
    bbox = np.asarray(bbox)
    # cut region takes xmin, xmax, ymin, ymax
    cut_faces, _ = cut_region(mesh, [bbox[0], bbox[2], bbox[1], bbox[3]])
    cut_faces = np.asarray(cut_faces)
    uniq, inv_index = np.unique(cut_faces.ravel(), return_inverse=True)
    new_faces = inv_index.reshape(cut_faces.shape)
    ret = xr_obj.isel(nod2=uniq)
    if isinstance(xr_obj, xr.DataArray):
        ret = ret.to_dataset()
    ret = ret.assign_coords({'faces': (('nelem', 'three'), new_faces)})
    return ret


def select_region(xr_obj: Union[xr.DataArray, xr.Dataset],
                  region: Region,
                  faces: Optional[Union[np.ndarray, xr.DataArray]] = None
                  ) -> xr.Dataset:
    from shapely.geometry import box, Polygon
    from shapely.prepared import prep
    from shapely.vectorized import contains as vectorized_contains

    if isinstance(region, Sequence) and len(region) == 4:
        region = box(*region)
    elif isinstance(region, Polygon):
        region = region
    else:
        raise NotImplementedError(f'Supplied region data {region} is not yet supported')

    faces = getattr(xr_obj, "faces", faces)
    if faces is None:
        raise ValueError(f"When passing a dataset it needs have faces in coords, or"
                         f"faces need to be passed explicitly.\n"
                         f"When passing a data array, argument faces can't be None,"
                         f"faces must be indices[nelem,3] that define triangles.")
    faces = np.asarray(faces)

    # buffer is necessry to facilitte floating point comparisions
    # buffer can be thought as tolerance around region in degrees
    # its value should be at least precision of data type of lats, lons (np.finfo)
    region = region.buffer(1e-6)
    prep_region = prep(region)
    selection = vectorized_contains(prep_region, np.asarray(xr_obj.lon), np.asarray(xr_obj.lat))
    if np.count_nonzero(selection) == 0:
        warnings.warn('No points in domain are within region, returning original data.')
        return xr_obj

    selection = selection[faces]
    face_mask = np.all(selection, axis=1)
    cut_faces = faces[face_mask]
    cut_faces = np.array(cut_faces, ndmin=1)
    uniq, inv_index = np.unique(cut_faces.ravel(), return_inverse=True)
    new_faces = inv_index.reshape(cut_faces.shape)
    ret = xr_obj.isel(nod2=uniq)

    if 'faces' in ret.coords:
        ret = ret.drop_vars('faces')

    if len(uniq) == 0:
        warnings.warn("No found points for the region are contained in dataset's triangulation (faces), "
                      "returning object without faces.")
        return ret  # no faces in coords

    if isinstance(xr_obj, xr.DataArray):
        ret = ret.to_dataset()

    ret = ret.assign_coords({'faces': (('nelem', 'three'), new_faces)})
    return ret


def select_points(xrobj: Union[xr.Dataset, xr.DataArray],
                  lon: Union[float, np.ndarray], lat: Union[float, np.ndarray], method='nearest', tolerance=None,
                  tree=None, return_distance=True, selection_dim_name="nod2", **other_dims) -> Union[
    xr.Dataset, xr.DataArray]:
    """
    TODO: check id all dims are of same length.
    """
    from cartopy.crs import Geocentric, Geodetic
    from scipy.spatial import cKDTree
    src_lons, src_lats = np.asarray(xrobj.lon), np.asarray(xrobj.lat)

    set_len_dims = {np.size(lon), np.size(lat), *[np.size(val) for val in other_dims.values()]}

    if len(set_len_dims) > 1:
        raise ValueError('For point selection length of all supplied dims args should be same.')

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

    other_dims = {k: xr.DataArray(np.array(v, ndmin=1), dims=selection_dim_name) for k, v in other_dims.items()}
    ret_obj = xrobj.isel(nod2=xr.DataArray(ind, dims=selection_dim_name)).sel(**other_dims, method=method)

    # from faces, which will not be useful in returned dataset
    # unless we reindex them, but is there a use case for that?
    if 'faces' in ret_obj.coords:
        ret_obj = ret_obj.drop_vars('faces')
    if return_distance:
        dist = distance_along_trajectory(lon, lat)
        dist_units, dist = normalize_distance(dist)
        ret_obj = ret_obj.assign_coords({'distance': (selection_dim_name, dist)})
        ret_obj.distance.attrs['units'] = dist_units
        ret_obj.distance.attrs['long_name'] = f"distance along trajectory"
    return ret_obj


def select(xrobj: Union[xr.Dataset, xr.DataArray], method='nearest',
           tolerance=None, region: Optional[Region] = None,
           path=Optional[Union[LineString, Sequence[float], MutableMapping]], tree=None,
           **indexers) -> Union[xr.Dataset, xr.DataArray]:
    """
    Higher level interface that does different kinds of selection emulates xarray's sel method.
    """
    lat = indexers.pop('lat', None)
    lon = indexers.pop('lon', None)
    lat_indexer = True if lat is not None else False
    lon_indexer = True if lon is not None else False

    if (lat_indexer or lon_indexer) and (region is not None or path is not None):
        # TODO: do this combinations better, doesn't check if path and region are both given
        raise Exception("Only one option: lat, lon as indexer or path or region is supported")

    ret_arr = xrobj

    if lat_indexer or lon_indexer:
        if lat_indexer and lon_indexer:
            if method == 'nearest':
                ret_arr = select_points(xrobj, lon, lat, method=method, tolerance=tolerance, tree=tree,
                                        return_distance=False)
            else:
                raise NotImplementedError("Only method='nearest' is currently supported.")
        else:
            raise NotImplementedError("Both lat, lon are needed as indexers, else use path or region.")
    elif region is not None:
        ret_arr = select_region(xrobj, region)
    elif path is not None:
        if isinstance(path, Sequence) or isinstance(path, LineString):
            if isinstance(path, LineString):
                path = np.asarray(path).T
            else:
                path = np.asarray(path)

            if not np.ndim(path) == 2:
                raise ValueError('Path of more then 2 columns (lons, lats) is ambiguous, use dictionary instead')
            else:
                lon, lat = path
                ret_arr = select_points(xrobj, lon, lat, method=method, tolerance=tolerance, tree=tree)
        elif isinstance(path, dict):
            ret_arr = select_points(xrobj, method=method, tolerance=tolerance, tree=tree, **path)
        else:
            raise ValueError('Invalid path argument it can only be sequence of (lons, lats), shapely 2D LineString or'
                             'dictionary containing coords.')

    # xarray doesn't support slice indexer when method argument is passed.
    # allow mixing indexers with values and slices.
    slice_indexers = {dim: dim_val for dim, dim_val in indexers.items() if isinstance(dim_val, slice)}

    if slice_indexers:
        ret_arr = ret_arr.sel(**slice_indexers)
        # remove slice indexers from indexers
        for dim in slice_indexers.keys():
            indexers.pop(dim)

    return ret_arr.sel(**indexers, method=method)


# Accessors

# Accessor Utils

def trimesh_plot(darr, tris, **other_dims):
    import geoviews as gv
    tris = np.asarray(tris)
    data = darr.sel(**other_dims)
    var_name = data.name
    var_da = gv.Dataset((darr.lon, darr.lat, data),
                        kdims=['lon', 'lat'], vdims=[var_name])
    return gv.TriMesh((tris, var_da))


# Dataset accessor

@xr.register_dataset_accessor("pyfesom2")
class FESOMDataset:
    def __init__(self, xr_dataset: xr.Dataset):
        self._xrobj = xr_obj = xr_dataset
        # TODO: check valid fesom data? otherwise accessor is available on all xarray datasets
        self._tree_obj = None
        for datavar in xr_obj.data_vars.keys():
            setattr(self, datavar, FESOMDataArray(xr_obj[datavar], xr_obj))

    def select(self, method='nearest', tolerance=None, region=None, path=None, **indexers):
        sel_obj = select(self._xrobj, method='nearest', tolerance=tolerance, region=region, path=path, **indexers)
        return sel_obj

    def select_points(self, lon: Union[float, np.ndarray], lat: Union[float, np.ndarray], method='nearest',
                      tolerance=None,
                      tree=None, return_distance=True, **other_dims):
        tree = self._tree
        return select_points(self._xrobj, lon, lat, method=method, tolerance=tolerance, tree=tree, **other_dims)

    def plot(self, *args, **kwargs):
        return self._xrobj.plot(*args, **kwargs)

    def triplot(self, *args, **kwargs):
        data = self._xrobj
        tri = Triangulation(data.lon, data.lat, triangles=data.faces)
        projection = kwargs.pop('projection', ccrs.PlateCarree())
        ax = kwargs.pop('ax', plt.axes(projection=projection))
        return ax.triplot(tri, *args, **kwargs)

    def _build_tree(self):
        from cartopy.crs import Geocentric, Geodetic
        from scipy.spatial import cKDTree
        geocentric_crs, geodetic_crs = Geocentric(), Geodetic()
        src_pts = geocentric_crs.transform_points(geodetic_crs, np.asarray(self._xrobj.lon),
                                                  np.asarray(self._xrobj.lat))
        self._tree_obj = cKDTree(src_pts, leafsize=32, compact_nodes=False, balanced_tree=False)
        return self._tree_obj

    @property
    def _tree(self):
        """Property to regulate tree access, _tree to hide from jupyter notebook"""
        if self._tree_obj is not None:
            return self._tree_obj
        return self._build_tree()

    def __repr__(self):
        return self._xrobj.__repr__()

    def _repr_html_(self):
        return self._xrobj._repr_html_()


class FESOMDataArray:
    """ A wrapper around Dataarray, that passes dataset context around"""

    def __init__(self, xr_dataarray: xr.DataArray, context_dataset=None):
        self._xrobj = xrobj = xr_dataarray
        self._context_dataset = context = context_dataset
        self._native_projection = ccrs.PlateCarree()

    def select(self, method='nearest', tolerance=None, region=None, path=None, **indexers):
        sel_obj = self._xrobj.to_dataset()
        sel_obj = sel_obj.assign_coords({'faces': (self._context_dataset.faces.dims,
                                                   self._context_dataset.faces.values)})
        tree = self._context_dataset.pyfesom2._tree
        sel_obj = select(sel_obj, method='nearest', tolerance=tolerance, region=region, path=path, tree=tree,
                         **indexers)
        return sel_obj

    def select_points(self, lon: Union[float, np.ndarray], lat: Union[float, np.ndarray], method='nearest',
                      tolerance=None,
                      tree=None, return_distance=True, **other_dims):
        tree = self._context_dataset.pyfesom2._tree
        return select_points(self._xrobj, lon, lat, method=method, tolerance=tolerance, tree=tree, **other_dims)

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
        non_plot_dims = {dim: self._xrobj[dim].values for dim in self._xrobj.dims if dim != 'nod2'}
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
        default_plot_types = ('line', 'contourf')  # 1d and 2d defaults
        extra_dims = [k for k in indexers_plot_kwargs.keys() if
                      k in self._xrobj.coords]  # coords is used instead of dims because xarray will raise error
        # in selection if user passes a invalid dim otherwise we would have to.

        extra_indexers = {dim: indexers_plot_kwargs.pop(dim) for dim in extra_dims}
        tree = self._context_dataset.pyfesom2._tree
        lon, lat = np.asarray(lon), np.asarray(lat)
        sel = select_points(self._xrobj, lon=lon, lat=lat, tree=tree, **extra_indexers)
        dim_len = len(sel.dims)  # determines plot type, 1d or 2d

        # make time as default bottom x-axis if present (else formatting gets hard)
        xax_dims = ('time', 'distance') if 'time' in extra_dims else ('distance', None)

        if 'time' in extra_dims:
            sel['points'] = sel.time
        else:
            sel['points'] = sel.distance

        sel = sel.transpose(..., 'points')  # push points to last for making it default xaxis for plots

        # use xarray plotting as it formats datetime axis with ease and adds sensible default to plot.
        if dim_len == 1:
            plot_type = default_plot_types[0] if plot_type == 'auto' else plot_type
            plot_fn = getattr(sel.plot, plot_type)
            plot = plot_fn(**indexers_plot_kwargs)[0]
        elif dim_len == 2:
            plot_type = default_plot_types[1] if plot_type == 'auto' else plot_type
            plot_fn = getattr(sel.plot, plot_type)
            plot = plot_fn(**indexers_plot_kwargs)
        else:
            raise Exception(f'A 2-d plot can have 1-2 dimensions, variable {sel.name} has dimensions: {sel.dims}.')

        ax = plot.ax if hasattr(plot, 'ax') else plot.axes  # akward some plots have it as ax, some as axes

        if xax_dims[1]:  # then it must be distance according to above.
            ax2 = ax.twiny()
            ax2.set_xlim(sel.distance.min(), sel.distance.max())
            ax2.set_xlabel(f'distance [{sel.distance.units}]')
        return plot

    def __repr__(self):
        return f"Wrapped {self._xrobj.__repr__()}\n{super().__repr__()}"

    def _repr_html_(self):
        return self._xrobj._repr_html_()
