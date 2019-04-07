#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pyfesom2` package."""

import pytest
import os
import numpy as np


from pyfesom2 import pyfesom2
from pyfesom2 import load_mesh
from pyfesom2 import get_data
from pyfesom2 import fesom2regular

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
    temp = get_data(data_path, 'temp', 1948, mesh)
    assert type(temp) == np.ndarray

    mmin = temp.min()
    assert mmin == pytest.approx(-1.8924446)

    mmax = temp.max()
    assert  mmax == pytest.approx(28.816469)
    
    # variable on elements
    u = get_data(data_path, 'u', 1948, mesh)
    assert type(u) == np.ndarray

    mmin = u.min()
    assert mmin == pytest.approx(-0.51486444)

    mmax = u.max()
    assert  mmax == pytest.approx(0.27181712)

    # 2d variable on vertices
    ice = get_data(data_path, 'a_ice', 1948, mesh)
    assert type(u) == np.ndarray

    mmean = ice.mean()
    assert mmean == pytest.approx(0.27451384)



def test_regriding():
    mesh_path = os.path.join(my_data_folder, 'pi-grid')
    data_path = os.path.join(my_data_folder, 'pi-results')
    mesh = load_mesh(mesh_path, usepickle = False, usejoblib = False)



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
