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
from pyfesom2 import get_meshdiag
from pyfesom2 import hovm_data
from pyfesom2 import select_depths
from pyfesom2 import volmean_data
from pyfesom2 import xmoc_data

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


def test_get_meshdiag():
    mesh_path = os.path.join(my_data_folder, "pi-grid")
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=False)
    diag = get_meshdiag(mesh)
    assert isinstance(diag, xr.Dataset)

    diag = get_meshdiag(mesh, meshdiag=os.path.join(mesh_path, "fesom.mesh.diag.nc"))
    assert isinstance(diag, xr.Dataset)


def test_hovm_data():
    mesh_path = os.path.join(my_data_folder, "pi-grid")
    data_path = os.path.join(my_data_folder, "pi-results")
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=False)

    # work on xarray
    # mean first, the hovm
    data = get_data(data_path, "temp", [1948, 1949], mesh, how="mean", compute=False)
    hovm = hovm_data(data, mesh)
    hovm_masked = hovm_data(data, mesh, mask=mesh.y2 > 65)
    assert hovm.shape == (1, 47)
    assert np.nanmean(hovm) == pytest.approx(7.446110751429013)
    assert np.nanmean(hovm_masked) == pytest.approx(1.5497698038738918)

    # hovm first, then mean
    data = get_data(data_path, "temp", [1948, 1949], mesh, how="ori", compute=False)
    hovm = hovm_data(data, mesh)
    assert hovm.shape == (2, 47)
    assert np.nanmean(hovm) == pytest.approx(7.446110751429013)

    # work on numpy array
    # mean first, the hovm
    data = get_data(data_path, "temp", [1948, 1949], mesh, how="mean", compute=True)
    hovm = hovm_data(data, mesh)
    hovm_masked = hovm_data(data, mesh, mask=mesh.y2 > 65)
    assert hovm.shape == (1, 47)
    assert np.nanmean(hovm) == pytest.approx(7.446110751429013)
    assert np.nanmean(hovm_masked) == pytest.approx(1.5497698038738918)

    # hovm first, then mean
    data = get_data(data_path, "temp", [1948, 1949], mesh, how="ori", compute=True)
    hovm = hovm_data(data, mesh)
    assert hovm.shape == (2, 47)
    assert np.nanmean(hovm) == pytest.approx(7.446110751429013)

    # test when only 1 time step of 3d field is in input
    data = get_data(data_path, "temp", [1948], mesh, how="mean", compute=False)
    hovm = hovm_data(data, mesh)
    assert hovm.shape == (1, 47)
    assert np.nanmean(hovm) == pytest.approx(7.440160989229884)

    data = get_data(data_path, "temp", [1948], mesh, how="mean", compute=True)
    hovm = hovm_data(data, mesh)
    assert hovm.shape == (1, 47)
    assert np.nanmean(hovm) == pytest.approx(7.440160989229884)

    # if we try to supply 2d variable
    with pytest.raises(ValueError):
        data = get_data(
            data_path, "a_ice", [1948, 1949], mesh, how="mean", compute=True
        )
        hovm = hovm_data(data, mesh)


def test_selec_depths():
    mesh_path = os.path.join(my_data_folder, "pi-grid")
    # data_path = os.path.join(my_data_folder, "pi-results")
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=False)

    assert select_depths(None, mesh) == range(0, 47)
    assert select_depths([0, "bottom"], mesh) == range(0, 47)
    assert select_depths([0, 100], mesh) == range(0, 12)
    assert select_depths([0, 10000], mesh) == range(0, 47)
    assert select_depths([0, 0], mesh) == range(0, 1)
    assert select_depths([500, 500], mesh) == range(20, 21)

def test_volmean_data():
    mesh_path = os.path.join(my_data_folder, "pi-grid")
    data_path = os.path.join(my_data_folder, "pi-results")
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=False)

    # xarray as input
    # time series
    data = get_data(data_path, "temp", [1948, 1949], mesh, how="ori", compute=False)
    result = volmean_data(data, mesh)
    result_masked = volmean_data(data, mesh, mask=mesh.y2 > 65)
    assert np.nanmean(result) == pytest.approx(3.3736459180632776)
    assert np.nanmean(result_masked) == pytest.approx(1.417763437740272)

    # one point
    data = get_data(data_path, "temp", [1948], mesh, how="ori", compute=False)
    result = volmean_data(data, mesh)
    assert np.nanmean(result) == pytest.approx(3.4039440198518953)

    data = get_data(data_path, "temp", [1948, 1949], mesh, how="mean", compute=False)
    result = volmean_data(data, mesh)
    assert np.nanmean(result) == pytest.approx(3.3736459180632776)

    # numpy array as input
    # time series
    data = get_data(data_path, "temp", [1948, 1949], mesh, how="ori", compute=True)
    result = volmean_data(data, mesh)
    result_masked = volmean_data(data, mesh, mask=mesh.y2 > 65)
    assert np.nanmean(result) == pytest.approx(3.3736459180632776)
    assert np.nanmean(result_masked) == pytest.approx(1.417763437740272)

    # one point
    data = get_data(data_path, "temp", [1948], mesh, how="ori", compute=True)
    result = volmean_data(data, mesh)
    assert np.nanmean(result) == pytest.approx(3.4039440198518953)

    data = get_data(data_path, "temp", [1948, 1949], mesh, how="mean", compute=True)
    result = volmean_data(data, mesh)
    assert np.nanmean(result) == pytest.approx(3.3736459180632776)

    # different depth ranges
    data = get_data(data_path, "temp", [1948], mesh, how="ori", compute=False)
    result = volmean_data(data, mesh, [0, 100])
    assert np.nanmean(result) == pytest.approx(16.26645462001221)

    data = get_data(data_path, "temp", [1948], mesh, how="ori", compute=False)
    result = volmean_data(data, mesh, [0, "bottom"])
    assert np.nanmean(result) == pytest.approx(3.4039440198518953)

    data = get_data(data_path, "temp", [1948], mesh, how="ori", compute=False)
    result = volmean_data(data, mesh, [500, 500])
    assert np.nanmean(result) == pytest.approx(6.339069486142839)


def test_xmoc_data():
    mesh_path = os.path.join(my_data_folder, "pi-grid")
    data_path = os.path.join(my_data_folder, "pi-results")
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=False)

    # xarray as input
    data = get_data(data_path, "w", [1948], mesh, how="mean", compute=False)
    lats, moc = xmoc_data(mesh, data)
    assert moc.mean() == pytest.approx(-5.283107985611987)
    assert moc.max() == pytest.approx(32.49306582121843)
    assert moc.min() == pytest.approx(-79.29266240207812)

    # numpy as input
    data = get_data(data_path, "w", [1948], mesh, how="mean", compute=True)
    lats, moc = xmoc_data(mesh, data)
    assert moc.mean() == pytest.approx(-5.283107985611987)
    assert moc.max() == pytest.approx(32.49306582121843)
    assert moc.min() == pytest.approx(-79.29266240207812)

    # masked
    # xarray as input
    data = get_data(data_path, "w", [1948], mesh, how="mean", compute=False)
    lats, moc = xmoc_data(mesh, data, mask="Atlantic_MOC")
    assert moc.mean() == pytest.approx(-6.4900976496575975)
    assert moc.max() == pytest.approx(19.88617032403639)
    assert moc.min() == pytest.approx(-49.8639074605277)

    # numpy as input
    data = get_data(data_path, "w", [1948], mesh, how="mean", compute=True)
    lats, moc = xmoc_data(mesh, data, mask="Atlantic_MOC")
    assert moc.mean() == pytest.approx(-6.4900976496575975)
    assert moc.max() == pytest.approx(19.88617032403639)
    assert moc.min() == pytest.approx(-49.8639074605277)

    # different nlats
    # xarray as input
    data = get_data(data_path, "w", [1948], mesh, how="mean", compute=False)
    lats, moc = xmoc_data(mesh, data, nlats=30)
    assert moc.mean() == pytest.approx(-5.5225348525691)
    assert moc.max() == pytest.approx(32.37976644734857)
    assert moc.min() == pytest.approx(-80.2349660966104)

    # numpy as input
    data = get_data(data_path, "w", [1948], mesh, how="mean", compute=True)
    lats, moc = xmoc_data(mesh, data, nlats=30)
    assert moc.mean() == pytest.approx(-5.5225348525691)
    assert moc.max() == pytest.approx(32.37976644734857)
    assert moc.min() == pytest.approx(-80.2349660966104)
