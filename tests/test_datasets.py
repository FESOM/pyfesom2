import pytest
import xarray as xr

from pyfesom2 import datasets


@pytest.fixture(scope="module", params=[datasets.core, datasets.arctic_1km, datasets.tutorial_dataset,
                                        datasets.rossby42, datasets.rossby42_level, datasets.rossby42_spatial])
def remote_dataset(request):
    da = request.param.load()
    yield da


@pytest.fixture(scope="module", params=[datasets.cmip6_lr, datasets.cmip6_mr, datasets.cmip6_hr])
def cmip6_grid(request):
    da = request.param.load()
    yield da


def check_dataset(dataset):
    assert isinstance(dataset, xr.Dataset)
    assert "nod2" in dataset.dims
    assert all([coord in dataset.coords for coord in ["lon", "lat"]])  # add faces eventually


@pytest.mark.xfail(reason="Remote dataset")
def test_remote_dataset(remote_dataset):
    check_dataset(remote_dataset)
    assert "Dataset URL" in remote_dataset.attrs


#@pytest.mark.xfail(reason="Local dataset issues")
#def test_local_dataset(local_dataset):
#    check_dataset(local_dataset)


@pytest.mark.xfail(reason="Remote dataset issues")
def test_cmip6_grids(cmip6_grid):
    assert isinstance(cmip6_grid, xr.Dataset)
    assert "ncells" in cmip6_grid.dims
    assert "depth" in cmip6_grid.dims
    assert "faces" in cmip6_grid.coords


def test_fesom_like():
    import pandas as pd
    ncells = 100
    ds = datasets.fesom_like(ncells)
    check_dataset(ds)
    assert len(ds.nod2) == ncells
    # test times parameter
    ncells, ntimes = 100, 10
    ds = datasets.fesom_like(ncells, times=ntimes)
    check_dataset(ds)
    assert len(ds.nod2) == ncells
    assert len(ds.time) == ntimes
    # check if we can pass datetime arrays
    times = pd.date_range('2001-01-01', freq='M', periods=ntimes)
    ds = datasets.fesom_like(ncells, times=times)
    assert len(ds.time) == len(times)

    ncells, ntimes, nlevels = 100, 10, 70
    ds = datasets.fesom_like(ncells, times=ntimes, levels=nlevels)
    check_dataset(ds)
    assert len(ds.nz1) == nlevels
    assert len(ds.nod2) == ncells
    assert len(ds.time) == ntimes
    # check if dims of data are in order of increasing size
    data_var = list(ds.data_vars.keys())[0]
    data_var_dims = ds[data_var].dims
    assert data_var_dims[0] == 'time'
    assert data_var_dims[1] == 'nz1'
    assert data_var_dims[2] == 'nod2'

    # check if level values can be passed
    levels = [0, -1.0, -2.0]  # or numpy array
    ds = datasets.fesom_like(ncells, times=ntimes, levels=levels)
    check_dataset(ds)
    assert len(ds.nz1) == len(levels)

    # check holes
    nholes=2
    prev_triangles = len(ds.nelem)
    ds = datasets.fesom_like(ncells, holes=nholes)
    check_dataset(ds)
