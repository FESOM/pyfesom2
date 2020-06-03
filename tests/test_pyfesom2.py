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
from pyfesom2 import fesom2regular
from pyfesom2 import plot

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

def test_regriding():
    mesh_path = os.path.join(my_data_folder, 'pi-grid')
    data_path = os.path.join(my_data_folder, 'pi-results')
    mesh = load_mesh(mesh_path, usepickle = False, usejoblib = False)
    lons = range(0,360)
    lats = range(-90,90)
    lons, lats = np.meshgrid(lons, lats)
    data = get_data(data_path, 'temp', 1948, mesh, depth=0)

    # default nn interpolation
    data_inter = fesom2regular(data, mesh, lons,lats)
    mmean = data_inter.mean()
    assert mmean == pytest.approx(6.309350409986763)
    assert isinstance(data_inter, np.ma.core.MaskedArray)

    # idist method
    data_inter = fesom2regular(data, mesh, lons,lats, how='idist')
    mmean = data_inter.mean()
    assert mmean == pytest.approx(6.308561066202526)
    assert isinstance(data_inter, np.ma.core.MaskedArray)

    # linear method from scipy
    data_inter = fesom2regular(data, mesh, lons,lats, how='linear')
    mmean = data_inter.mean()
    assert mmean == pytest.approx(13.582933890477655)
    assert isinstance(data_inter, np.ma.core.MaskedArray)

    # cubic method from scipy
    data_inter = fesom2regular(data, mesh, lons,lats, how='cubic')
    mmean = data_inter.mean()
    assert mmean == pytest.approx(13.39402615207708)
    assert isinstance(data_inter, np.ma.core.MaskedArray)

    # clean up
    os.remove(os.path.join(mesh_path, 'distances_3140_0_359_-90_89_360_180_1'))
    os.remove(os.path.join(mesh_path, 'distances_3140_0_359_-90_89_360_180_5'))

    os.remove(os.path.join(mesh_path, 'inds_3140_0_359_-90_89_360_180_1'))
    os.remove(os.path.join(mesh_path, 'inds_3140_0_359_-90_89_360_180_5'))

    os.remove(os.path.join(mesh_path, 'qhull_3140'))

# @pytest.mark.skip(reason="slow")
def test_plot():
    mesh_path = os.path.join(my_data_folder, 'pi-grid')
    data_path = os.path.join(my_data_folder, 'pi-results')
    figure_path = os.path.join(my_data_folder, 'baseline_images')
    mesh = load_mesh(mesh_path, abg=[50, 15, -90], usepickle = False, usejoblib = False)
    data = get_data(data_path, 'temp', 1948, mesh, depth=0)

    # standard plot
    ax = plot(mesh,data, influence=800000)
    plt.savefig('./out.png')
    baseline_image = os.path.join(figure_path, 'plot_temp_basic.png')
    compare_images('./out.png', baseline_image, tol=10)
    os.remove('./out.png')

    # inverce distance interpolation
    ax = plot(mesh,data, influence=800000, interp='idist')
    plt.savefig('./out.png')
    baseline_image = os.path.join(figure_path, 'plot_temp_idist.png')
    compare_images('./out.png', baseline_image, tol=10)
    os.remove('./out.png')

    # pc projection
    ax = plot(mesh,data, influence=800000, mapproj = 'pc')
    plt.savefig('./out.png')
    baseline_image = os.path.join(figure_path, 'plot_temp_pc.png')
    compare_images('./out.png', baseline_image, tol=10)
    os.remove('./out.png')

    # pc projection
    ax = plot(mesh,data, influence=800000, mapproj = 'pc')
    plt.savefig('./out.png')
    baseline_image = os.path.join(figure_path, 'plot_temp_pc.png')
    compare_images('./out.png', baseline_image, tol=10)
    os.remove('./out.png')

    # np projection
    ax = plot(mesh,data, influence=800000, mapproj = 'np')
    plt.savefig('./out.png')
    baseline_image = os.path.join(figure_path, 'plot_temp_np.png')
    compare_images('./out.png', baseline_image, tol=10)
    os.remove('./out.png')

    #assert isinstance(fig, matplotlib.figure.Figure)

@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
