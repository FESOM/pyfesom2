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

from collections.abc import Sequence  # python>3.3?
from typing import Union, Sequence as SequenceType, Tuple

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.tri import Triangulation
from shapely.geometry import MultiPoint, MultiPolygon, Polygon, box

# New Types
BoundingBox = SequenceType[float]
MultiRegion = Union[SequenceType[Polygon], MultiPolygon]


# Selection functions
# TODO: Probably function names should be prefixed with spatial.

def select_points(xrobj: Union[xr.Dataset, xr.DataArray], lon, lat, method='nearest', tolerance=None, tree=None):
    """
    Selects points from dataset or data array.

    Selection is based on tree
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


def select_region(xrobj: Union[xr.Dataset, xr.DataArray], region: Union[BoundingBox, Polygon]) -> xr.DataArray:
    """
    Selects data for a region. region can be specified as a Bounding Box [West, Soutn, North, East ] or
    as shapely polygons or for multi region selection: a list of Polygons or MultiPolygon.

    This uses xarray's standard methods on stacked data
    TODO: handle multi polygons as list or Multipolygon
    Parameters
    ----------
    region

    Can be bounding box or shapely polygon

    Returns
    -------
    xr.DataArray
    """
    # import pandas as pd
    if isinstance(region, Sequence) and len(region) == 4:
        sel_polygon = box(*region)
    elif isinstance(region, Polygon):
        sel_polygon = region
    else:
        raise NotImplementedError(f'Supplied region data {region} is not yet supported')
    data = xrobj
    # if not isinstance(xrobj.indexes['nod2'], pd.MultiIndex):
    if 'nod2' not in data.indexes:  # cheaper way then importing pandas
        data = xrobj.set_index(nod2=['lon', 'lat'])

    mesh_as_mp = MultiPoint(data.nod2.values.T)  # should we save it for subsequent selections?, profile this.
    # self._mesh_as_mp = asMultiPoint(self.stacked_data.nod2.values.T) # faster but net is slower why?, profile this

    # selection by indexes # in efficient because of list comprehension although isel is efficient.
    # inds = [i for i, pt in enumerate(mesh_as_mp) if pt.within(sel_polygon)]
    # #inds = list(set(inds)) # there are a few grids with dulplicates
    # return data.isel(nod2=inds)

    sel_indexer = np.asarray(mesh_as_mp.intersection(sel_polygon))
    sel_indexer = list((lo, la) for lo, la in sel_indexer)
    # sel_indexer = list(set((lo, la) for lo, la in sel_indexer))  # removing duplicates

    if len(sel_indexer) == 0:
        raise Exception('No region found')

    return data.sel(nod2=sel_indexer)


# Accessors

@xr.register_dataset_accessor("pyfesom2")
class FESOMDataset:
    def __init__(self, xr_dataset: xr.Dataset):
        self._xrobj = xr_obj = xr_dataset
        # TODO: check valid fesom data?
        self._tree = None
        for datavar in xr_obj.data_vars:
            # setattr(self, str(datavar), xr_obj[datavar])
            setattr(self, str(datavar), FESOMDataArray2(xr_obj[datavar], xr_obj))

    def select(self):
        pass

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


class FESOMDataArray2:
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
        return ax.triplot(tri, *args, **kwargs)

    def trimesh(self, levels=None, cmap='RdBu', colorbar=True, height=400, width=600,
                colorbar_position="bottom", tools=['hover']):
        import geoviews as gv
        import holoviews as hv
        from holoviews.operation.datashader import rasterize
        import numpy as np
        hv.extension('bokeh')

        data = self._xrobj
        var_name = data.name
        projection, tri = self._triangulate(data, ccrs.PlateCarree())
        tris = gv.Dataset(tri.triangles, kdims=['v0', 'v1', 'v2'])
        verts = gv.Dataset((tri.x, tri.y, data), kdims=['lon', 'lat'], vdims=[var_name])
        plot = gv.TriMesh((tris, verts))
        plot_opts = {}
        if isinstance(levels, int):
            plot_opts.update({'color_levels':levels})
        elif isinstance(levels, list):
            if len(levels)>2:
                plot_opts.update({'color_levels':len(levels)})
                plot = plot.redim.range(var_name=(min(levels),max(levels)))
            elif len(levels) == 2:
                plot = plot.redim.range(var_name=levels)
        elif levels is None:
            pass
        else:
            raise Exception('Invalid levels')


        return rasterize(plot).opts(tools=tools, width=width,
                                    height=height, projection=projection,
                                    cmap=cmap, colorbar=colorbar, **plot_opts)

@xr.register_dataarray_accessor("pyfesom2")
class FESOMDataArray(object):
    def __init__(self, xr_dataarray):
        self._xrobj = xr_dataarray
        self._xrobj_stacked = None
        self._mesh_as_mp = None
        self._tree = None

    @property
    def stacked_data(self):
        if not self._xrobj_stacked:
            if "nod2" in self._xrobj.dims:
                self._xrobj_stacked = self._xrobj.set_index(nod2=['lon', 'lat'])
            else:
                self._xrobj_stacked = self._xrobj.stack(nod2=['lon', 'lat'])
        return self._xrobj_stacked

    def select_region(self, region: Union[BoundingBox, Polygon]) -> xr.DataArray:
        """
        Selects data for a region. region can be specified as a Bounding Box [West, Soutn, North, East ] or
        as shapely polygons or for multi region selection: a list of Polygons or MultiPolygon.
        TODO: handle multi polygons as list or Multipolygon
        Parameters
        ----------
        region

        Returns
        -------
        xr.DataArray
        """

        if isinstance(region, Sequence) and len(region) == 4:
            sel_polygon = box(*region)
        elif isinstance(region, Polygon):
            sel_polygon = region
        else:
            raise NotImplementedError(f'Supplied region data {region} is not yet supported')

        if self._mesh_as_mp is None:
            self._mesh_as_mp = MultiPoint(self.stacked_data.nod2.values.T)
            # self._mesh_as_mp = asMultiPoint(self.stacked_data.nod2.values.T) # faster

        # sel by index
        # inds = [i for i, pt in enumerate(self._mesh_as_mp) if pt.within(sel_polygon)]
        # #inds = list(set(inds)) # there are a few grids with dulplicates
        # return self.stacked_data.isel(nod2=inds)

        sel_indexer = np.asarray(self._mesh_as_mp.intersection(sel_polygon))
        sel_indexer = list((lo, la) for lo, la in sel_indexer)
        # sel_indexer = list(set((lo, la) for lo, la in sel_indexer))  # removing duplicates

        if len(sel_indexer) == 0:
            raise Exception('No region found')

        return self._xrobj_stacked.sel(nod2=sel_indexer)

    def select_point(self, lon, lat, method='nearest', tolerance=None):
        from cartopy.crs import Geocentric, Geodetic
        from scipy.spatial import cKDTree
        if not method == 'nearest':
            raise NotImplementedError("Spatial selection currently supports only nearest neighbor lookup")
        geocentric_crs, geodetic_crs = Geocentric(), Geodetic()
        if self._tree is None:
            src_pts = geodetic_crs.transform_points(geodetic_crs, self._xrobj.lon.values, self._xrobj.lat.values)
            self._tree = cKDTree(src_pts)
        dst_pts = geodetic_crs.transform_points(geodetic_crs, np.array(lon), np.array(lat))

        if tolerance is None:
            # query is faster when no tolerance is specified
            _, ind = self._tree.query(dst_pts)
        else:
            inds = self._tree.query_ball_point(dst_pts, r=tolerance, n_jobs=-1)[0]
            # find min dist
            ind = inds
        return self._xrobj.isel(nod2=ind)

    def sel(self, method='nearest', tolerance=None, region=None, path=None, **indexers):
        """
        TODO: support advanced xarray indexing
        Parameters
        ----------
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

        if not method == "nearest":
            raise NotImplementedError("Only nearest method is currently supported")
        if (lat_indexer or lon_indexer) and (region is not None or path is not None):
            # TODO: do this combinations better, doesn't check if path and region are both given
            raise Exception("Onlu one option: lat, lon as indexer or path or region is supported")

        if lat_indexer or lon_indexer:
            if lat_indexer and lon_indexer:
                ret_arr = self.select_point(lon, lat, method=method, tolerance=tolerance)
            else:
                raise NotImplementedError("Both lat, lon are needed as indexer, else use path or region")
        elif region is not None:
            ret_arr = self.select_region(region)
        elif path is not None:
            raise NotADirectoryError('Path option is not implemented yet')
        else:
            print('in else')
            ret_arr = self._xrobj

        return ret_arr.sel(**indexers, method=method)

    def isel(self, **indexers):
        return self._xrobj.isel(**indexers)

    def plot_map(self, *args, **kwargs):
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
        from matplotlib.tri import Triangulation

        data = self._xrobj.squeeze()
        if len(data.dims) > 1 or "nod2" not in data.dims:
            raise Exception('Not a spatial dataset')

        projection = kwargs.pop('projection', ccrs.PlateCarree())
        ax = kwargs.pop('ax', plt.axes(projection=projection))

        minv, maxv = data.min().values, data.max().values
        tri = Triangulation(data.lon, data.lat, data.faces)
        data = data.fillna(minv - 9999)  # make sure it is out of data bounds
        levels = kwargs.pop('levels', np.linspace(minv, maxv, 100))
        colorbar = kwargs.pop('colorbar', True)
        pl = ax.tripcolor(tri, data.values.ravel(), levels=levels, transform=ccrs.PlateCarree(), *args, **kwargs)

        if colorbar:
            plt.colorbar(pl, ax=ax)
        return pl

    def plot_transect(self):
        pass

    def regrid(self):
        pass

    def regrid_like(self):
        raise NotImplementedError
