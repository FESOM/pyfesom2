import warnings
from typing import Optional, Sequence, Union, MutableMapping

import cartopy.crs as ccrs
import numpy as np
import xarray as xr
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
        raise ValueError(f"When passing a dataset it needs have faces in coords, or "
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
        raise ValueError(f"Supplied region data can be a sequence of (minlon, minlat, maxlon, maxlat) or "
                                  f"a Shapely's Polygon. This {region} is not supported.")

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
        raise ValueError("Only one option: lat, lon as indexer or path or region is supported")

    ret_arr = xrobj

    if lat_indexer or lon_indexer:
        if lat_indexer and lon_indexer:
            if method == 'nearest':
                ret_arr = select_points(xrobj, lon, lat, method=method, tolerance=tolerance, tree=tree,
                                        return_distance=False)
            else:
                raise NotImplementedError("Only method='nearest' is currently supported.")
        else:
            raise ValueError("Both lat, lon are needed as indexers, else use path, region arguments or "
                                      ".select_points(lon=..., lat=...) method.")
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
        sel_obj = select(self._xrobj, method=method, tolerance=tolerance, region=region, path=path, **indexers)
        return sel_obj

    def select_points(self, lon: Union[float, np.ndarray], lat: Union[float, np.ndarray], method='nearest',
                      tolerance=None,
                      tree=None, return_distance=True, **other_dims):
        tree = self._tree
        return select_points(self._xrobj, lon, lat, method=method, tolerance=tolerance, tree=tree, **other_dims)

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

    def __repr__(self):
        return f"Wrapped {self._xrobj.__repr__()}\n{super().__repr__()}"

    def _repr_html_(self):
        return self._xrobj._repr_html_()
