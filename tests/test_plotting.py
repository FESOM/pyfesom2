#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pyfesom2` package."""

import pytest
import os
import numpy as np
import xarray as xr
from matplotlib import cm
import matplotlib.pylab as plt
from matplotlib.testing.compare import compare_images
from matplotlib.testing.decorators import _image_directories


from pyfesom2 import load_mesh
from pyfesom2 import get_data
from pyfesom2 import create_proj_figure
from pyfesom2 import get_plot_levels
from pyfesom2 import plot
from pyfesom2 import get_vector_forplot

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
my_data_folder = os.path.join(THIS_DIR, "data")


def test_create_proj_figure():
    fig, ax = create_proj_figure("pc", (1, 1), (10, 10))
    assert ax.get_ylim()[0] == 0
    assert ax.get_ylim()[1] == 1
    fig, ax = create_proj_figure("merc", (1, 1), (10, 10))


def test_get_plot_levels():
    mesh_path = os.path.join(my_data_folder, "pi-grid")
    data_path = os.path.join(my_data_folder, "pi-results")
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=False)
    data = get_data(data_path, "temp", 1948, mesh, depth=0)
    data_levels = get_plot_levels([0, 1, 5], data, lev_to_data=False)
    assert data_levels[3] == 0.75
    data_levels = get_plot_levels([0, 1, 5, 10], data, lev_to_data=False)
    assert data_levels[3] == 10
    data_levels = get_plot_levels([-100, 100, 5], data, lev_to_data=True)
    assert data_levels[3] == pytest.approx(21.345652550458908)
    data_levels = get_plot_levels([-100, 100, 5], data, lev_to_data=False)
    assert data_levels[3] == 50


@pytest.mark.skip(reason="slow")
def test_plot():
    mesh_path = os.path.join(my_data_folder, "pi-grid")
    data_path = os.path.join(my_data_folder, "pi-results")
    figure_path = os.path.join(my_data_folder, "baseline_images")
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=False)
    data = get_data(data_path, "temp", 1948, mesh, depth=0)

    # standard plot
    ax = plot(mesh, data, influence=800000)
    plt.savefig("./out.png")
    baseline_image = os.path.join(figure_path, "plot_temp_basic.png")
    results = compare_images("./out.png", baseline_image, tol=10)
    assert results is None
    os.remove("./out.png")

    # inverce distance interpolation
    ax = plot(mesh, data, influence=800000, interp="idist")
    plt.savefig("./out.png")
    baseline_image = os.path.join(figure_path, "plot_temp_idist.png")
    results = compare_images("./out.png", baseline_image, tol=10)
    assert results is None
    os.remove("./out.png")

    # pc projection
    ax = plot(mesh, data, influence=800000, mapproj="pc")
    plt.savefig("./out.png")
    baseline_image = os.path.join(figure_path, "plot_temp_pc.png")
    results = compare_images("./out.png", baseline_image, tol=10)
    assert results is None
    os.remove("./out.png")

    # np projection
    ax = plot(mesh, data, influence=800000, mapproj="np", box=[-180, 180, 60, 90])
    plt.savefig("./out.png")
    baseline_image = os.path.join(figure_path, "plot_temp_np.png")
    results = compare_images("./out.png", baseline_image, tol=10)
    assert results is None
    os.remove("./out.png")


def test_get_transect_uv():

    mesh_path = os.path.join(my_data_folder, "pi-grid")
    mesh = load_mesh(mesh_path, usepickle=False, usejoblib=False)
    data_path = os.path.join(my_data_folder, "pi-results")
    u = get_data(data_path, "u", [1948, 1949], mesh, compute=False, depth=0)
    v = get_data(data_path, "v", [1948, 1949], mesh, compute=False, depth=0)

    u_int, v_int, lonreg2, latreg2 = get_vector_forplot(u, v, mesh)
    assert u_int.max() == pytest.approx(0.22456094548268307)
    assert u_int.min() == pytest.approx(-0.43932501398679424)
    assert u_int.mean() == pytest.approx(0.003800295094638076)
    assert v_int.max() == pytest.approx(0.2895587461946785)
    assert v_int.min() == pytest.approx(-0.30255398462888544)
    assert v_int.mean() == pytest.approx(0.0009492933726720353)

    u = get_data(data_path, "u", [1948, 1949], mesh, compute=False, depth=1000)
    v = get_data(data_path, "v", [1948, 1949], mesh, compute=False, depth=1000)

    u_int, v_int, lonreg2, latreg2 = get_vector_forplot(u, v, mesh)
    assert u_int.max() == pytest.approx(0.09659441223255835)
    assert u_int.min() == pytest.approx(-0.0856496846437692)
    assert u_int.mean() == pytest.approx(0.002234351731582896)
    assert v_int.max() == pytest.approx(0.09343932803578202)
    assert v_int.min() == pytest.approx(-0.10588274761901853)
    assert v_int.mean() == pytest.approx(-0.00017386484505706984)