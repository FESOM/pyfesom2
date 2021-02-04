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
# TODO: Probably function names should be prefixed with spatial.

def select_points(xrobj: Union[xr.Dataset, xr.DataArray],
                  lon, lat, method='nearest', tolerance=None, tree=None) -> Union[xr.Dataset, xr.DataArray]:
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

    """
    from cartopy.crs import Geocentric, Geodetic
    from scipy.spatial import cKDTree
    if not method == 'nearest':
        raise NotImplementedError("Spatial selection currently supports only nearest neighbor lookup")
    geocentric_crs, geodetic_crs = Geocentric(), Geodetic()
    if tree is None:
        src_pts = geodetic_crs.transform_points(geodetic_crs, xrobj.lon.values, xrobj.lat.values)
        tree = cKDTree(src_pts)
    dst_pts = geodetic_crs.transform_points(geodetic_crs, np.array(lon), np.array(lat))

    if tolerance is None:
        # query is faster when no tolerance is specified
        _, ind = tree.query(dst_pts)
    else:
        inds = tree.query_ball_point(dst_pts, r=tolerance, n_jobs=-1)[0]
        # find min dist
        ind = inds
    return xrobj.isel(nod2=ind)


class SimpleMesh:
    """Wrapper that fakes pyfesom's mesh for purposes of this module"""

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
                if tolerance:
                    raise NotImplementedError('tolerance is not supported for method: nearest, try method:'
                                              'projected_nearest')
                ret_arr = sel_nn_points(xrobj, lon, lat)
            elif method == 'projected_nearest':
                ret_arr = select_points(xrobj, lon, lat, method=method, tolerance=tolerance)
            else:
                raise NotImplementedError("Only methods 'nearest, projected_nearest' are currently supported")
        else:
            raise NotImplementedError("Both lat, lon are needed as indexers, else use path or region")
    elif region is not None:
        ret_arr = select_region(xrobj, region)
    elif path is not None:
        raise NotImplementedError('Path option is not implemented yet')
    else:
        ret_arr = xrobj

    return ret_arr.sel(**indexers, method=method)


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

def trimesh_plot(darr,tris, time, nz1):
    import geoviews as gv
    tris = np.asarray(tris)
    data = darr.sel(time=time,nz1=nz1)
    var_name = data.name
    var_da =  gv.Dataset((darr.lon,darr.lat,data),
                         kdims=['lon','lat'], vdims=[var_name])
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
                colorbar_position="bottom", projection=None, tools=['hover'], **hv_kwopts):
        try:
            import geoviews as gv
            import holoviews as hv
            from holoviews.operation.datashader import rasterize
            hv.extension('bokeh')
        except ImportError as ex:
            raise ImportError('Using trimesh needs geoviews[holoviews, bokeh] and datashader') from ex

        trimesh_fn = functools.partial(trimesh_plot, darr=self._xrobj, tris=self._context_dataset.faces)
        var_name = self._xrobj.name
        hvd = hv.Dataset(self._xrobj.drop_vars(['lon','lat']))
        dmap = hv.DynamicMap(trimesh_fn,
                             kdims=hvd.kdims).redim.values(nz1=self._xrobj.nz1.values,
                                                           time=self._xrobj.time.values)
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
        return rasterize(dmap).opts(width=width,
                                    height=height,
                                    cmap=cmap, colorbar=colorbar, tools=tools, **plot_opts)

    def select(self, method='nearest', tolerance=None, region=None, path=None, **indexers):
        sel_obj = select(self._xrobj, method='nearest', tolerance=None, region=None, path=None, **indexers)
        return sel_obj.to_dataset()

