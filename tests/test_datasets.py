from pyfesom2 import datasets
import xarray as xr


def test_lcore():
    lcore_dataset_dict = datasets.datasets_dict['LCORE']
    lcore_dataset = datasets.LCORE.load()
    assert isinstance(lcore_dataset, xr.Dataset)
    assert all([var in lcore_dataset.data_vars for var in lcore_dataset_dict['vars']])

def test_a01():
    a01_dataset = datasets.A01.load()
    assert isinstance(lcore_dataset, xr.Dataset)

