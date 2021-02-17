import numpy as np
import pytest
from shapely.geometry import box

from pyfesom2.datasets import LCORE, open_dataset


# TODO: skipping A01 because chunksize is too small, leading to slow sel
@pytest.fixture(scope='module', params=[LCORE])  # , A01])
def lcore_dataset(request):
    da = request.param.load()
    yield da


# push this to test config so that it is available globally
@pytest.fixture(scope='module')
def local_dataset(request):
    import os.path
    cur_dir = os.path.dirname(request.fspath)
    data_path = os.path.join(cur_dir, "data", "pi-results", "temp.fesom.*.nc")
    mesh_path = os.path.join(cur_dir, "data", "pi-grid")
    da = open_dataset(data_path, mesh_path=mesh_path)
    yield da


# use scope function so that we can change grid size dynamically,
# but merged dataset needs to be changed appropriately
@pytest.fixture(scope='module', params=[36 * 18, 360 * 180])  # same pts as 10deg, 1deg reg grid
def random_dataset(request):
    import xarray as xr
    from matplotlib.tri import Triangulation
    if request.param is not None:
        data_size = request.param
    else:
        data_size = 36 * 18  # default, it covers more edge cases
    lons = np.random.uniform(-180., 180., data_size)
    lats = np.random.uniform(-90., 90., data_size)
    tris = Triangulation(lons, lats)
    dataset = xr.Dataset(coords={'lon'  : ('nod2', lons),
                                 'lat'  : ('nod2', lats),
                                 'faces': (('nelem', 'three'), tris.triangles)})
    dataset['dummy_var'] = ('nod2', np.random.uniform(0., 1., data_size))

    yield dataset


# merged dataset fixture
@pytest.fixture(params=["local_dataset", "random_dataset"])
def dataset(local_dataset, random_dataset, request):
    yield request.getfixturevalue(request.param)


# def test_dataarray_accessor(dataset):
#     vars = list(dataset.data_vars.keys())
#     # check if data vars exist on accessor
#     for data_var in vars:
#          assert hasattr(dataset.pyfesom2, data_var), f"data var {data_var} is anot available from pyfesom2 accessor"
#     # check if region selection works
#     # bounding box selection using list
#     bbox = [0., 60., 60., 90.] # West, South, East, North ( LL -> UR )
#     box_sel = getattr(dataset.pyfesom2,vars[0]).select(region=bbox)
#     assert isinstance(box_sel, xr.DataArray), "Region selection using bbox list failed"
#     # TODO: check if bounds match
#     # Polygon selection
#     bbox = box(*bbox)  # make a polygon from box
#     box_sel = dataset[vars[0]].pyfesom2.select_region(region=bbox)
#     assert isinstance(box_sel, xr.DataArray), "Region selection using Polygon failed"
#     # Point selection
#     point_sel = dataset[vars[0]].pyfesom2.select_point(lon=0, lat=0)
#     assert isinstance(point_sel, xr.DataArray)
#
#     point_sel = dataset[vars[0]].pyfesom2.select_point(lon=0, lat=0, tolerance=10)
#     assert isinstance(point_sel, xr.DataArray)
#
#     with pytest.raises(NotImplementedError):
#         point_sel = dataset[vars[0]].pyfesom2.select_point(lon=0, lat=0, method=None)
#
#     sel_arr = dataset[vars[0]].pyfesom2.sel(lat=0, lon=0)
#     assert isinstance(sel_arr, xr.DataArray)
#     assert len(sel_arr.nod2) == 1
#
#     sel_arr = dataset[vars[0]].pyfesom2.sel(region=bbox)
#     assert isinstance(sel_arr, xr.DataArray)
#
#     if "time" in dataset.dims:
#         mintime, maxtime = dataset.time.min(), dataset.time.max()
#         sel_arr = dataset[vars[0]].pyfesom2.sel(time=mintime, region=bbox)


def test_distance_along_trajectory():
    """ Check if cumsum of n linerly spaced points at constant latitude is distance to one*n-1_
    """
    from pyfesom2.accessor import distance_along_trajectory
    n = 100
    lons = np.linspace(-10, 180, 100)
    lats = np.zeros(lons.shape, dtype=lons.dtype)  # at equator
    dists = distance_along_trajectory(lons, lats)
    print(dists, dists.shape)
    assert np.isclose(dists[0], 0.), "Distance to first point must always be zero."
    print(dists[1], dists[-1], dists[1] * (n - 1))
    assert np.isclose(dists[1] * (n - 1), dists[-1]), "Total distance along linearly spaced trajectory doesn't " \
                                                      "match (n-1)*individual distance."


def test_normalize_distances():
    from pyfesom2.accessor import normalize_distance

    dists_lt_1km = np.linspace(0, 999, 10)
    dists_gt_1km = np.linspace(1000, 10000, 2)
    units, dist = normalize_distance(np.hstack([dists_lt_1km, dists_gt_1km]))
    assert units == 'm'
    assert ~np.any(np.isnan(dist)), 'NaNs in dists should not be possible.'

    dists_lt_1km = np.linspace(0, 999, 2)
    dists_gt_1km = np.linspace(1000, 10000, 10)
    units, dist = normalize_distance(np.hstack([dists_lt_1km, dists_gt_1km]))
    assert units == 'km'
    assert ~np.any(np.isnan(dist)), 'NaNs in dists should not be possible.'


bbox_tests = [
    (-10, 50, 60, 80),  # LL -> UR (minx, miny, maxx, maxy)
    # use bad values and check
    # bounds smaller then grid size
    # invalid bounds.
]


@pytest.mark.parametrize("bbox", bbox_tests)
def test_select_bbox(dataset, bbox):
    from pyfesom2.accessor import select_bbox

    # selection on datasets
    sda = select_bbox(dataset, bbox)
    slon_min, slat_min, slon_max, slat_max = (sda.lon.min(), sda.lat.min(),
                                              sda.lon.max(), sda.lon.max())
    # check within bbox's bounds
    # because pyfesom2.ut's cut_region uses closed bounds and sel is only when all nodes
    # of triangle are in bounds.

    assert slon_min >= bbox[0] and slon_max <= bbox[2]
    assert slat_min >= bbox[1] and slat_max <= bbox[3]

    # selection on data arrays
    data_var = list(dataset.data_vars.keys())[0]
    with pytest.raises(ValueError):
        select_bbox(dataset[data_var], bbox)

    sda = select_bbox(dataset[data_var], bbox, faces=dataset.faces)
    slon_min, slat_min, slon_max, slat_max = (sda.lon.min(), sda.lat.min(),
                                              sda.lon.max(), sda.lon.max())
    assert slon_min >= bbox[0] and slon_max <= bbox[2]
    assert slat_min >= bbox[1] and slat_max <= bbox[3]


region_tests = [
    *bbox_tests,  # bbox tests should be valid inputs for select_region
    # *[box(*bbox) for bbox in bbox_tests],  # bbox tests cst s shapely's box
    # Polygon([(-70, 30), (-10, 0), (-10, 60)]),  # a triangle in atlantic
    # in future add complex region
    # polygons from pyfesom's ut
]


@pytest.mark.parametrize("region", region_tests)
def test_select_region(dataset, region):
    from typing import Sequence
    import warnings
    from shapely.geometry import MultiPoint
    from pyfesom2.accessor import select_region
    #     # convex hull of returned dataset
    if isinstance(region, Sequence):
        outer_polygon = box(*region)
    else:
        outer_polygon = region

    sda = select_region(dataset, region)

    if len(sda.lon) == 0:
        with pytest.warns(UserWarning):
            warnings.warn("Found no points for the region in the domain.", UserWarning)
        assert not hasattr(sda, "faces")

    mp = MultiPoint(np.vstack((sda.lon, sda.lat)).T)
    print(outer_polygon, mp.convex_hull)
    assert outer_polygon.contains(mp.convex_hull)


@pytest.mark.parametrize("npoints", [10])
def test_select_points(dataset, npoints, request):
    from pyfesom2.accessor import select_points
    # dataset = local_dataset # makes it easy to change fixture
    # select random points from datapoints
    random_pts = np.random.choice(len(dataset.nod2), npoints)  # replace=True by default
    lats = dataset.lat[random_pts]
    lons = dataset.lon[random_pts]

    # pass lons and lats as numpy array
    sda = select_points(dataset, lons.values, lats.values)
    assert np.array_equal(sda.lon.values, lons.values) & np.array_equal(sda.lat.values, lats.values)

    # test passing lon and lat as scalars
    slon, slat = np.array(lons.values[0], ndmin=1), np.array(lats.values[0], ndmin=1)  # to match shapes
    sda = select_points(dataset, float(slon), float(slat))
    assert np.array_equal(sda.lon.values, slon) & np.array_equal(sda.lat.values, slat)

    # selection on data arrays
    data_var = list(dataset.data_vars.keys())[0]
    sda = select_points(dataset[data_var], lons, lats)
    assert np.array_equal(sda.lon.values, lons) & np.array_equal(sda.lat.values, lats)

    # pass lons and lats as xarray data array
    sda = select_points(dataset, lons, lats)
    assert np.array_equal(sda.lon.values, lons) & np.array_equal(sda.lat.values, lats)

    # check other attrs
    assert 'points' in sda.dims, 'point selection will always have points in dimensions'
    assert 'faces' not in sda, 'faces are not returned by select points'

    assert 'distance' in sda.coords, 'by default select points returns distance as coordinate'
    assert np.isclose(sda.distance[0], 0.0), ' distance to first point is always 0.'

    sda = select_points(dataset, lons, lats, return_distance=False)
    assert 'distance' not in sda, ' point selection with return_distance=False does not have distance'
