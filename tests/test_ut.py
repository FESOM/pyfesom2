#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pyfesom2` package."""

import pytest
import os
import numpy as np
import xarray as xr

from pyfesom2 import get_mask
from pyfesom2 import compute_face_coords
from pyfesom2 import load_mesh
from pyfesom2 import cut_region

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
my_data_folder = os.path.join(THIS_DIR, "data")

def test_get_mask():
    mesh_path = os.path.join(my_data_folder, "pi-grid")
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=False)

    mask = get_mask(mesh, "Amerasian basin")
    assert mask.shape[0] == 361
    assert mask.max() == 1172
    assert mask.min() == 0

    mask = get_mask(mesh, "Atlantic_Basin")
    assert mask.shape[0] == 3140
    assert mask.dtype == np.dtype('bool')

def test_compute_face_coords():
    mesh_path = os.path.join(my_data_folder, "pi-grid")
    mesh = load_mesh(mesh_path, abg=[50, 15, -90], usepickle=False, usejoblib=False)

    face_x, face_y = compute_face_coords(mesh)
    assert face_x.min() == -179.89608131062724
    assert face_x.max() == 179.9390442333396
    assert face_x.mean() == 5.192477589299442

    assert face_y.min() == -77.84857205391366
    assert face_y.max() == 88.69953826328107
    assert face_y.mean() == 9.148515091596915

def test_cut_region():
    mesh_path = os.path.join(my_data_folder, "pi-grid")
    mesh = load_mesh(mesh_path, abg=[50, 15, -90], usepickle=False, usejoblib=False)
    elem_no_nan = cut_region(mesh, box=[0, 30, 70, 85])
    assert elem_no_nan.min() == 161
    assert elem_no_nan.max() == 742
    assert elem_no_nan.mean() == pytest.approx(382.5257270693512)


