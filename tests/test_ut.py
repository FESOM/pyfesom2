#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pyfesom2` package."""

import pytest
import os
import numpy as np
import xarray as xr
from matplotlib import cm

from pyfesom2 import get_mask
from pyfesom2 import compute_face_coords
from pyfesom2 import load_mesh
from pyfesom2 import cut_region
from pyfesom2 import get_cmap
from pyfesom2 import get_no_cyclic

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
my_data_folder = os.path.join(THIS_DIR, "data")


def test_get_mask():
    mesh_path = os.path.join(my_data_folder, "pi-grid")
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=False)

    mask = get_mask(mesh, "Amerasian basin")
    assert mask.shape[0] == 148
    assert mask.max() == 1292
    assert mask.min() == 749

    mask = get_mask(mesh, "Atlantic_Basin")
    assert mask.shape[0] == 3140
    assert mask.dtype == np.dtype("bool")


def test_compute_face_coords():
    mesh_path = os.path.join(my_data_folder, "pi-grid")
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=False)

    face_x, face_y = compute_face_coords(mesh)
    assert face_x.min() == pytest.approx(-179.89608133333334)
    assert face_x.max() == pytest.approx(179.93904426666666)
    assert face_x.mean() == pytest.approx(5.192477589615796)

    assert face_y.min() == pytest.approx(-77.84857205333334)
    assert face_y.max() == pytest.approx(88.69953826333334)
    assert face_y.mean() == pytest.approx(9.148515091582464)


def test_cut_region():
    mesh_path = os.path.join(my_data_folder, "pi-grid")
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=False)
    elem_no_nan, no_nan_triangles = cut_region(mesh, box=[0, 30, 70, 85])
    assert elem_no_nan.min() == 161
    assert elem_no_nan.max() == 742
    assert elem_no_nan.mean() == pytest.approx(382.5257270693512)
    assert len(elem_no_nan) == 149
    assert np.sum(no_nan_triangles) == 149
    assert len(no_nan_triangles) == 5839


def test_get_cmap():
    colormap = get_cmap("Spectral_r")
    assert colormap.name == "Spectral_r"
    colormap = get_cmap("thermal")
    assert colormap.name == "thermal"
    colormap = get_cmap()
    assert colormap.name == "Spectral_r"
    colormap = get_cmap(cm.Accent)
    assert colormap.name == "Accent"


def test_get_no_cyclic():
    mesh_path = os.path.join(my_data_folder, "pi-grid")
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=False)

    no_cyclic_elem = get_no_cyclic(mesh, mesh.elem)
    no_cyclic_elem = np.array(no_cyclic_elem)
    assert no_cyclic_elem.max() == 5838
    assert no_cyclic_elem.min() == 0
    assert no_cyclic_elem.mean() == pytest.approx(2907.772830452244)
    assert no_cyclic_elem.shape[0] == 5727
