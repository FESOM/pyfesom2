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
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=False)

    face_x, face_y = compute_face_coords(mesh)
    assert face_x.min() == -179.85463333333334
    assert face_x.max() == 179.95329999999998
    assert face_x.mean() == -7.086726174573275

    assert face_y.min() == -83.60326666666664
    assert face_y.max() == 85.14420000000001
    assert face_y.mean() == 9.647414751384375

