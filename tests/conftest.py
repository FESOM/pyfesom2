import pytest
from pyfesom2.datasets import open_dataset

#TODO: scope should be session and autouse=True
@pytest.fixture(scope="function", autouse=False)
def local_dataset(request):
    import os.path
    cur_dir = os.path.dirname(__file__)#request.fspath)
    data_path = os.path.join(cur_dir, "data", "pi-results", "*.nc")
    mesh_path = os.path.join(cur_dir, "data", "pi-grid")
    da = open_dataset(data_path, mesh_path=mesh_path)
    da.load()
    da.close() 
    yield da
