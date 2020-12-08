from pyfesom2 import datasets
import xarray as xr
import pytest



@pytest.fixture(scope="module", params=[datasets.LCORE, datasets.A01])
def remote_dataset(request):
    da = request.param.load()
    yield da

@pytest.fixture
def local_dataset(request):
    import os.path
    cur_dir = os.path.dirname(request.fspath)
    data_path = os.path.join(cur_dir, "data", "pi-results", "*.nc")
    mesh_path = os.path.join(cur_dir, "data", "pi-grid")
    da = datasets.open_dataset(data_path, mesh_path=mesh_path)
    yield da

def check_dataset(dataset):
    assert isinstance(dataset, xr.Dataset)
    assert "nod2" in dataset.dims
    assert all([coord in dataset.coords for coord in ["lon", "lat"]])


def test_remote_dataset(remote_dataset):
    check_dataset(remote_dataset)
    assert "Dataset URL" in remote_dataset.attrs

def test_local_dataset(local_dataset):
    check_dataset(local_dataset)
