from pyfesom2 import datasets
import xarray as xr
import pytest


@pytest.fixture(scope="module", params=[datasets.LCORE, datasets.A01])
def dataset(request):
    da = request.param.load()
    yield da


def test_dataset(dataset):
    assert isinstance(dataset, xr.Dataset)
    assert "nod2" in dataset.dims
    assert all([coord in dataset.coords for coord in ["lon", "lat"]])
    assert "Dataset URL" in dataset.attrs
