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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
my_data_folder = os.path.join(THIS_DIR, "data")

def test_create_proj_figure():
    fig, ax = create_proj_figure('pc', (1,1), (10,10))
    assert ax.get_ylim()[0] == 0
    assert ax.get_ylim()[1] == 1
    fig, ax = create_proj_figure('merc', (1,1), (10,10))


def test_get_plot_levels():
    mesh_path = os.path.join(my_data_folder, "pi-grid")
    data_path = os.path.join(my_data_folder, "pi-results")
    mesh = load_mesh(mesh_path, abg=[50, 15, -90], usepickle=False, usejoblib=False)
    data = get_data(data_path, "temp", 1948, mesh, depth=0)
    data_levels = get_plot_levels([0,1,5], data, lev_to_data=False)
    assert data_levels[3] ==  0.75
    data_levels = get_plot_levels([0,1,5, 10], data, lev_to_data=False)
    assert data_levels[3] ==  10
    data_levels =  get_plot_levels([-100, 100,  5], data, lev_to_data=True)
    assert data_levels[3] ==  pytest.approx(21.13924074)
    data_levels =  get_plot_levels([-100, 100,  5], data, lev_to_data=False)
    assert data_levels[3] == 50

# @pytest.mark.skip(reason="slow")
def test_plot():
    mesh_path = os.path.join(my_data_folder, 'pi-grid')
    data_path = os.path.join(my_data_folder, 'pi-results')
    figure_path = os.path.join(my_data_folder, 'baseline_images')
    mesh = load_mesh(mesh_path, abg=[50, 15, -90], usepickle = False, usejoblib = False)
    data = get_data(data_path, 'temp', 1948, mesh, depth=0)

    # standard plot
    ax = plot(mesh,data, influence=800000)
    plt.savefig('./out.png')
    baseline_image = os.path.join(figure_path, 'plot_temp_basic.png')
    compare_images('./out.png', baseline_image, tol=10)
    os.remove('./out.png')

    # inverce distance interpolation
    ax = plot(mesh,data, influence=800000, interp='idist')
    plt.savefig('./out.png')
    baseline_image = os.path.join(figure_path, 'plot_temp_idist.png')
    compare_images('./out.png', baseline_image, tol=10)
    os.remove('./out.png')

    # pc projection
    ax = plot(mesh,data, influence=800000, mapproj = 'pc')
    plt.savefig('./out.png')
    baseline_image = os.path.join(figure_path, 'plot_temp_pc.png')
    compare_images('./out.png', baseline_image, tol=10)
    os.remove('./out.png')

    # pc projection
    ax = plot(mesh,data, influence=800000, mapproj = 'pc')
    plt.savefig('./out.png')
    baseline_image = os.path.join(figure_path, 'plot_temp_pc.png')
    compare_images('./out.png', baseline_image, tol=10)
    os.remove('./out.png')

    # np projection
    ax = plot(mesh,data, influence=800000, mapproj = 'np')
    plt.savefig('./out.png')
    baseline_image = os.path.join(figure_path, 'plot_temp_np.png')
    compare_images('./out.png', baseline_image, tol=10)
    os.remove('./out.png')

    #assert isinstance(fig, matplotlib.figure.Figure)

