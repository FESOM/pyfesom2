#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pyfesom2` package."""

import pytest
import os
import numpy as np
import xarray as xr

from pyfesom2 import pyfesom2
from pyfesom2 import load_mesh
from pyfesom2 import get_data
from pyfesom2 import ice_ext
from pyfesom2 import ice_vol
from pyfesom2 import ice_area

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
my_data_folder = os.path.join(THIS_DIR, "data")


def test_ice_integrals():
    mesh_path = os.path.join(my_data_folder, "pi-grid")
    data_path = os.path.join(my_data_folder, "pi-results")
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=False)

    # default get_data (with how='mean') should work.

    data = get_data(data_path, "a_ice", 1948, mesh, depth=0, compute=False)
    ext = ice_ext(data, mesh)
    assert ext.data[0] == pytest.approx(12710587600895.246)

    data = get_data(data_path, "a_ice", 1948, mesh, depth=0, compute=False)
    area = ice_area(data, mesh)

    assert area.data[0] == pytest.approx(9066097785122.738)

    data = get_data(data_path, "m_ice", 1948, mesh, depth=0, compute=False)
    vol = ice_vol(data, mesh)

    assert vol.data[0] == pytest.approx(13403821068217.506)

    # work with xarray as input
    data = get_data(data_path, "a_ice", 1948, mesh, depth=0, how="ori", compute=False)
    ext = ice_ext(data, mesh)
    area = ice_area(data, mesh)

    assert ext.data[0] == pytest.approx(12710587600895.246)
    assert area.data[0] == pytest.approx(9066097785122.738)

    data = get_data(data_path, "m_ice", 1948, mesh, depth=0, how="ori", compute=False)
    vol = ice_vol(data, mesh)
    assert vol.data[0] == pytest.approx(13403821068217.506)

    # work with numpy array as input
    data = get_data(data_path, "a_ice", 1948, mesh, depth=0, how="ori", compute=True)
    ext = ice_ext(data, mesh)

    # have to load data once again, since ice_ext actually modify numpy array.
    # I don't want to add .copy to the `ice_ext` function.
    data = get_data(data_path, "a_ice", 1948, mesh, depth=0, how="ori", compute=True)
    area = ice_area(data, mesh)

    assert ext.data[0] == pytest.approx(12710587600895.246)
    assert area.data[0] == pytest.approx(9066097785122.738)

    data = get_data(data_path, "m_ice", 1948, mesh, depth=0, how="ori", compute=True)
    vol = ice_vol(data, mesh)
    assert vol.data[0] == pytest.approx(13403821068217.506)

