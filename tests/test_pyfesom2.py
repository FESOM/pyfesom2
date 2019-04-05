#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pyfesom2` package."""

import pytest
import os


from pyfesom2 import pyfesom2
from pyfesom2 import load_mesh

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
my_data_folder = os.path.join(THIS_DIR, 'data')


def test_readmesh():
    mesh_path = os.path.join(my_data_folder, 'pi-grid')
    mesh = load_mesh(mesh_path, usepickle = False, usejoblib = False)
    assert mesh.n2d == 3140
    assert mesh.e2d == 5839
    
    print(mesh)
    

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
