## support sel on remote
## and opened dataset
from typing import Union, List, Sequence as SequenceType
import numpy as np
import xarray as xr
from shapely.geometry import MultiPoint, MultiPolygon, Polygon, box, asMultiPoint

from collections.abc import Sequence # python>3.3?


# FESOMDataset


# New Types
BoundingBox = SequenceType[float]
MultiRegion = Union[SequenceType[Polygon], MultiPolygon]


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
            #self._mesh_as_mp = asMultiPoint(self.stacked_data.nod2.values.T) # faster

        # sel by index
        # inds = [i for i, pt in enumerate(self._mesh_as_mp) if pt.within(sel_polygon)]
        # #inds = list(set(inds)) # there are a few grids with dulplicates
        # return self.stacked_data.isel(nod2=inds)


        sel_indexer = np.asarray(self._mesh_as_mp.intersection(sel_polygon))
        sel_indexer = list((lo, la) for lo, la in sel_indexer)
        #sel_indexer = list(set((lo, la) for lo, la in sel_indexer))  # removing duplicates

        if len(sel_indexer)==0:
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
            #TODO: do this combinations better, doesn't check if path and region are both given
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

    def plot_map(self):
        pass

    def plot_transect(self):
        pass

    def regrid(self):
        pass

    def regrid_like(self):
        raise NotImplementedError

