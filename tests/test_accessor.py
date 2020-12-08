from pyfesom2.datasets import LCORE, A01
from pyfesom2.accessor import FESOMDataArray
import pytest
from shapely.geometry import box
import xarray as xr

#TODO: skipping A01 because chunksize is too small, leading to slow sel
@pytest.fixture(scope='module', params=[LCORE])  #, A01])
def dataset(request):
    da = request.param.load()
    yield da


def test_dataarray_accessor(dataset):
    vars = list(dataset.data_vars.keys())
    # check if accessor exists
    assert hasattr(dataset[vars[0]], "pyfesom2"), "dataarray doesn't have accessor pyfesom2"
    # check if region selection works
    # bounding box selection using list
    bbox = [0.,60., 60., 90.]
    box_sel = dataset[vars[0]].pyfesom2.select_region(region=bbox)
    assert isinstance(box_sel, xr.DataArray), "Region selection using bbox list failed"
    # TODO: check if bounda match
    # Polygon selection
    bbox = box(*bbox) # Polygon
    box_sel = dataset[vars[0]].pyfesom2.select_region(region=bbox)
    assert isinstance(box_sel, xr.DataArray), "Region selection using Polygon failed"
    # Point selection
    point_sel = dataset[vars[0]].pyfesom2.select_point(lon=0,lat=0)
    assert isinstance(point_sel, xr.DataArray)

    point_sel = dataset[vars[0]].pyfesom2.select_point(lon=0,lat=0, tolerance=10)
    assert isinstance(point_sel, xr.DataArray)

    with pytest.raises(NotImplementedError):
        point_sel = dataset[vars[0]].pyfesom2.select_point(lon=0, lat=0, method=None)

    sel_arr = dataset[vars[0]].pyfesom2.sel(lat=0, lon=0)
    assert isinstance(sel_arr, xr.DataArray)
    assert len(sel_arr.nod2) == 1

    sel_arr = dataset[vars[0]].pyfesom2.sel(region=bbox)
    assert isinstance(sel_arr, xr.DataArray)

    if "time" in dataset.dims:
        mintime,maxtime = dataset.time.min(), dataset.time.max()
        sel_arr = dataset[vars[0]].pyfesom2.sel(time=mintime, region=bbox)

