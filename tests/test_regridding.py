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
from pyfesom2 import fesom2regular
from pyfesom2 import tonodes
from pyfesom2 import tonodes3d

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
my_data_folder = os.path.join(THIS_DIR, "data")


def test_fesom2regular():
    mesh_path = os.path.join(my_data_folder, "pi-grid")
    data_path = os.path.join(my_data_folder, "pi-results")
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=False)
    lons = range(0, 360)
    lats = range(-90, 90)
    lons, lats = np.meshgrid(lons, lats)
    data = get_data(data_path, "temp", 1948, mesh, depth=0)

    # default nn interpolation
    data_inter = fesom2regular(data, mesh, lons, lats)
    assert data_inter.mean() == pytest.approx(5.547857817853418)
    assert isinstance(data_inter, np.ma.core.MaskedArray)

    # idist method
    data_inter = fesom2regular(data, mesh, lons, lats, how="idist")
    assert data_inter.mean() == pytest.approx(5.547480765409886)
    assert isinstance(data_inter, np.ma.core.MaskedArray)

    # linear method from scipy
    data_inter = fesom2regular(data, mesh, lons, lats, how="linear")
    assert data_inter.mean() == pytest.approx(13.492437572309811)
    assert isinstance(data_inter, np.ma.core.MaskedArray)

    # cubic method from scipy
    data_inter = fesom2regular(data, mesh, lons, lats, how="cubic")
    assert data_inter.mean() == pytest.approx(13.42669663949152)
    assert isinstance(data_inter, np.ma.core.MaskedArray)

    # clean up
    os.remove(os.path.join(mesh_path, "distances_3140_0_359_-90_89_360_180_1"))
    os.remove(os.path.join(mesh_path, "distances_3140_0_359_-90_89_360_180_5"))

    os.remove(os.path.join(mesh_path, "inds_3140_0_359_-90_89_360_180_1"))
    os.remove(os.path.join(mesh_path, "inds_3140_0_359_-90_89_360_180_5"))

    os.remove(os.path.join(mesh_path, "qhull_3140"))


def test_tonodes():
    mesh_path = os.path.join(my_data_folder, "pi-grid")
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=False)
    data_path = os.path.join(my_data_folder, "pi-results")
    u = get_data(data_path, "u", [1948, 1949], mesh, depth=0)
    v = get_data(data_path, "v", [1948, 1949], mesh, depth=0)

    u_nodes = tonodes(u, mesh)
    v_nodes = tonodes(v, mesh)

    assert u_nodes.min() == pytest.approx(-0.438402373837165)
    assert u_nodes.mean() == pytest.approx(-0.0022651651208314854)
    assert u_nodes.max() == pytest.approx(0.21826703478530526)
    assert v_nodes.min() == pytest.approx(-0.30760535729567123)
    assert v_nodes.mean() == pytest.approx(0.0030737476137769277)
    assert v_nodes.max() == pytest.approx(0.2809557692901927)

    # now with xarray
    u = get_data(data_path, "u", [1948, 1949], mesh, depth=0, compute=False)
    v = get_data(data_path, "v", [1948, 1949], mesh, depth=0, compute=False)

    u_nodes = tonodes(u, mesh)
    v_nodes = tonodes(v, mesh)

    assert u_nodes.min() == pytest.approx(-0.438402373837165)
    assert u_nodes.mean() == pytest.approx(-0.0022651651208314854)
    assert u_nodes.max() == pytest.approx(0.21826703478530526)
    assert v_nodes.min() == pytest.approx(-0.30760535729567123)
    assert v_nodes.mean() == pytest.approx(0.0030737476137769277)
    assert v_nodes.max() == pytest.approx(0.2809557692901927)


def test_tonodes3d():
    mesh_path = os.path.join(my_data_folder, "pi-grid")
    mesh = load_mesh(mesh_path, abg=[50, 15, -90], usepickle=False, usejoblib=False)
    data_path = os.path.join(my_data_folder, "pi-results")
    u = get_data(data_path, "u", [1948, 1949], mesh)
    v = get_data(data_path, "v", [1948, 1949], mesh)

    u_nodes = tonodes3d(u, mesh)
    v_nodes = tonodes3d(v, mesh)

    assert u_nodes.shape == (3140, 47)
    assert v_nodes.shape == (3140, 47)

    assert u_nodes.min() == pytest.approx(-0.438402373837165)
    assert u_nodes.mean() == pytest.approx(0.0014840021120303578)
    assert u_nodes.max() == pytest.approx(0.21826703478530526)
    assert v_nodes.min() == pytest.approx(-0.30760535729567123)
    assert v_nodes.mean() == pytest.approx(0.0003354015441015618)
    assert v_nodes.max() == pytest.approx(0.2813401806588615)

    # now with xarray
    u = get_data(data_path, "u", [1948, 1949], mesh, compute=False)
    v = get_data(data_path, "v", [1948, 1949], mesh, compute=False)

    u_nodes = tonodes3d(u, mesh)
    v_nodes = tonodes3d(v, mesh)

    assert u_nodes.shape == (3140, 47)
    assert v_nodes.shape == (3140, 47)

    assert u_nodes.min() == pytest.approx(-0.438402373837165)
    assert u_nodes.mean() == pytest.approx(0.0014840021120303578)
    assert u_nodes.max() == pytest.approx(0.21826703478530526)
    assert v_nodes.min() == pytest.approx(-0.30760535729567123)
    assert v_nodes.mean() == pytest.approx(0.0003354015441015618)
    assert v_nodes.max() == pytest.approx(0.2813401806588615)

