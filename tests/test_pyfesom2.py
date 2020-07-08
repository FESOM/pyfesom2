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
    mesh = load_mesh(mesh_path, usepickle = False, usejoblib = False)
    assert mesh.n2d == 3140
    assert mesh.e2d == 5839
    mesh = load_mesh(mesh_path, usepickle = True, usejoblib = False)
    assert os.path.exists(os.path.join(my_data_folder, 'pi-grid', 'pickle_mesh_py3_fesom2'))
    os.remove(os.path.join(my_data_folder, 'pi-grid', 'pickle_mesh_py3_fesom2'))
    mesh = load_mesh(mesh_path, usepickle = False, usejoblib = True)
    assert os.path.exists(os.path.join(my_data_folder, 'pi-grid', 'joblib_mesh_fesom2'))
    os.remove(os.path.join(my_data_folder, 'pi-grid', 'joblib_mesh_fesom2'))
    mesh = load_mesh(mesh_path)
    assert os.path.exists(os.path.join(my_data_folder, 'pi-grid', 'pickle_mesh_py3_fesom2'))
    os.remove(os.path.join(my_data_folder, 'pi-grid', 'pickle_mesh_py3_fesom2'))
    print(mesh)

def test_get_data():
    mesh_path = os.path.join(my_data_folder, 'pi-grid')
    data_path = os.path.join(my_data_folder, 'pi-results')
    mesh = load_mesh(mesh_path, usepickle = False, usejoblib = False)
    # variable on vertices
    temp = get_data(data_path, 'temp', 1948, mesh, depth=0)
    assert type(temp) == np.ndarray

    mmin = temp.min()
    assert mmin == pytest.approx(-1.8924446)

    mmax = temp.max()
    assert  mmax == pytest.approx(28.816469)

    # variable on elements
    u = get_data(data_path, 'u', 1948, mesh, depth=0)
    assert type(u) == np.ndarray

    mmin = u.min()
    assert mmin == pytest.approx(-0.51486444)

    mmax = u.max()
    assert  mmax == pytest.approx(0.27181712)

    # 2d variable on vertices
    ice = get_data(data_path, 'a_ice', 1948, mesh, depth=0)
    assert type(u) == np.ndarray

    mmean = ice.mean()
    assert mmean == pytest.approx(0.27451384)

    # get multiple years
    temp = get_data(data_path, 'temp', [1948, 1949], mesh, depth=0)
    mmean = temp.mean()
    assert mmean == pytest.approx(8.5541878)

    # get one record from multiple files
    temp = get_data(data_path, 'temp', [1948, 1949], mesh, records=slice(0,1), depth=0)
    mmean = temp.mean()
    assert mmean == pytest.approx(8.4580183)

    # get different depth
    temp = get_data(data_path, 'temp', [1948, 1949], mesh, depth = 200)
    mmean = temp.mean()
    assert mmean == pytest.approx(5.3856239)

    # get different depth and different aggregation
    temp = get_data(data_path, 'temp', [1948, 1949], mesh, depth = 200, how='max')
    mmean = temp.mean()
    assert mmean == pytest.approx(5.6487503)

    # directly open ncfile (in data 1948, but we directly request 1949)
    temp = get_data(data_path, 'temp', [1948], mesh, depth = 200, how='max',
                    ncfile='{}/{}'.format(data_path, "temp.fesom.1949.nc"))
    mmean = temp.mean()
    assert mmean == pytest.approx(5.3057818)

    # return dask array
    temp = get_data(data_path, 'temp', [1948, 1949], mesh, depth = 200, how='max', 
                    compute=False)
    assert isinstance(temp, xr.DataArray)

    # use range as argument
    temp = get_data(data_path, 'temp', range(1948, 1950), mesh, depth = 200, how='max')
    mmean = temp.mean()
    assert mmean == pytest.approx(5.6487503)
