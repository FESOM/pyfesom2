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
from pyfesom2 import transect_get_lonlat
from pyfesom2 import get_transect

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
my_data_folder = os.path.join(THIS_DIR, "data")


def test_get_transect():

    mesh_path = os.path.join(my_data_folder, "pi-grid")
    mesh = load_mesh(mesh_path, abg=[50, 15, -90], usepickle=False, usejoblib=False)
    data_path = os.path.join(my_data_folder, "pi-results")
    data = get_data(data_path, "temp", [1948, 1949], mesh, compute=False)

    lon_start = -149
    lat_start = 70.52
    lon_end = -149
    lat_end = 73

    lonlat = transect_get_lonlat(lon_start, lat_start, lon_end, lat_end)
    dist, transect_data = get_transect(data, mesh, lonlat)
    assert dist.shape[0] == 30
    assert dist.max() == pytest.approx(258.87327988009883)
    assert dist.min() == 0
    assert transect_data.shape == (30, 47)
    assert transect_data.min() == pytest.approx(-1.446849)
    assert transect_data.max() == pytest.approx(0.38322645)



