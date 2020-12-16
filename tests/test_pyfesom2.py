#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pyfesom2` package."""

import pytest
import os
import numpy as np
import xarray as xr
import matplotlib.pylab as plt
# import matplotlib
from matplotlib.testing.compare import compare_images
from matplotlib.testing.decorators import _image_directories


from pyfesom2 import pyfesom2
from pyfesom2 import load_mesh
from pyfesom2 import get_data


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
my_data_folder = os.path.join(THIS_DIR, 'data')


def test_readmesh():
    mesh_path = os.path.join(my_data_folder, 'pi-grid')
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=False)
    assert mesh.n2d == 3140
    assert mesh.e2d == 5839
    mesh = load_mesh(mesh_path, usepickle=True, usejoblib=False)
    assert os.path.exists(os.path.join(my_data_folder, 'pi-grid', 'pickle_mesh_py3_fesom2'))
    os.remove(os.path.join(my_data_folder, 'pi-grid', 'pickle_mesh_py3_fesom2'))
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=True)
    assert os.path.exists(os.path.join(my_data_folder, 'pi-grid', 'joblib_mesh_py3_fesom2'))
    os.remove(os.path.join(my_data_folder, 'pi-grid', 'joblib_mesh_py3_fesom2'))
    mesh = load_mesh(mesh_path)
    assert os.path.exists(os.path.join(my_data_folder, 'pi-grid', 'pickle_mesh_py3_fesom2'))
    os.remove(os.path.join(my_data_folder, 'pi-grid', 'pickle_mesh_py3_fesom2'))
    print(mesh)


def test_get_data():
    mesh_path = os.path.join(my_data_folder, 'pi-grid')
    data_path = os.path.join(my_data_folder, 'pi-results')
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=False)
    # variable on vertices
    temp = get_data(data_path, 'temp', 1948, mesh, depth=0)
    assert type(temp) == np.ndarray

    assert temp.min() == pytest.approx(-1.8680784)
    assert temp.max() == pytest.approx(29.083563)

    # variable on elements
    u = get_data(data_path, 'u', 1948, mesh, depth=0)
    assert type(u) == np.ndarray

    assert u.min() == pytest.approx(-0.5859177)
    assert u.max() == pytest.approx(0.30641124)

    # 2d variable on vertices
    ice = get_data(data_path, 'a_ice', 1948, mesh, depth=0)
    assert type(u) == np.ndarray

    assert ice.mean() == pytest.approx(0.2859408)

    # get multiple years
    temp = get_data(data_path, 'temp', [1948, 1949], mesh, depth=0)
    assert temp.mean() == pytest.approx(8.664016)

    # get one record from multiple files
    temp = get_data(data_path, 'temp', [1948, 1949], mesh, records=slice(0, 1), depth=0)
    assert temp.mean() == pytest.approx(8.670743)

    # get different depth
    temp = get_data(data_path, 'temp', [1948, 1949], mesh, depth=200)
    assert temp.mean() == pytest.approx(6.2157564)

    # get different depth and different aggregation
    temp = get_data(data_path, 'temp', [1948, 1949], mesh, depth=200, how='max')
    assert temp.mean() == pytest.approx(6.3983703)

    # directly open ncfile (in data 1948, but we directly request 1949)
    temp = get_data(data_path, 'temp', [1948], mesh, depth=200, how='max',
                    ncfile='{}/{}'.format(data_path, "temp.fesom.1949.nc"))
    assert temp.mean() == pytest.approx(6.2478514)

    # return dask array
    temp = get_data(data_path, 'temp', [1948, 1949], mesh, depth = 200, how='max', 
                    compute=False)
    assert isinstance(temp, xr.DataArray)

    # use range as argument
    temp = get_data(data_path, 'temp', range(1948, 1950), mesh, depth = 200, how='max')
    mmean = temp.mean()
    assert mmean == pytest.approx(6.3983703)
