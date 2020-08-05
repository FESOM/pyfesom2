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
from pyfesom2 import get_transect_uv
from pyfesom2 import tonodes3d

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
my_data_folder = os.path.join(THIS_DIR, "data")


def test_get_transect():

    mesh_path = os.path.join(my_data_folder, "pi-grid")
    mesh = load_mesh(mesh_path, abg=[50, 15, -90], usepickle=False, usejoblib=False)
    data_path = os.path.join(my_data_folder, "pi-results")
    data = get_data(data_path, "temp", [1948, 1949], mesh)

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

def test_get_transect_uv():

    mesh_path = os.path.join(my_data_folder, "pi-grid")
    mesh = load_mesh(mesh_path, abg=[50, 15, -90], usepickle=False, usejoblib=False)
    data_path = os.path.join(my_data_folder, "pi-results")
    u = get_data(data_path, "u", [1948, 1949], mesh)
    v = get_data(data_path, "v", [1948, 1949], mesh)

    u_nodes = tonodes3d(u, mesh)
    v_nodes = tonodes3d(v, mesh)

    lon_start = 120
    lat_start = 75
    lon_end = 120
    lat_end = 80
    lonlat = transect_get_lonlat(lon_start, lat_start, lon_end, lat_end, 30)

    dist, rot_u, rot_v = get_transect_uv(u_nodes, v_nodes, mesh, lonlat, myangle=0)

    assert dist.shape == (30.,)
    assert rot_u.shape == (30,47)
    assert rot_v.shape == (30,47)
    assert rot_u.max() == pytest.approx(0.012118130919170302)
    assert rot_u.min() == pytest.approx(-0.010595729181566588)
    assert rot_v.max() == pytest.approx(0.014261195850716241)
    assert rot_v.min() == pytest.approx(-0.0024032835331608696)

    dist, rot_u_90, rot_v_90 = get_transect_uv(u_nodes, v_nodes, mesh, lonlat, myangle=90)

    assert rot_u_90.max() == pytest.approx(0.014261195850716241)
    assert rot_u_90.min() == pytest.approx(-0.0024032835331608696)
    assert rot_v_90.max() == pytest.approx(0.012118130919170302) 
    assert rot_v_90.min() == pytest.approx(-0.010595729181566588)



