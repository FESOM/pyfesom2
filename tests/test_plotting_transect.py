#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `pyfesom2` package's transect and cross-section plotting functionality."""

import os
import pytest
import numpy as np
import xarray as xr
import math
from pathlib import Path
from unittest import mock
from typing import List, Tuple, Union, Optional

import matplotlib.pyplot as plt

# Get the directory of the current file
THIS_DIR = Path(__file__).parent.absolute()
TEST_DATA_DIR = THIS_DIR / 'data'
PI_GRID_PATH = TEST_DATA_DIR / 'pi-grid'

@pytest.fixture
def mock_cartopy():
    """Mock cartopy modules for testing."""
    with mock.patch('pyfesom2.plotting.CARTOPY_AVAILABLE', True):
        with mock.patch('pyfesom2.plotting.ccrs') as mock_ccrs:
            # Configure mock projection objects
            for proj in ['Mercator', 'PlateCarree', 'NorthPolarStereo', 'SouthPolarStereo', 'Robinson']:
                proj_mock = mock.MagicMock()
                getattr(mock_ccrs, proj).return_value = proj_mock
            yield mock_ccrs

@pytest.fixture
def mock_mesh():
    """Create a mock mesh object for testing."""
    mesh = mock.MagicMock()
    # Configure mesh properties
    mesh.x2 = np.array([0, 10, 20, 30, 40])
    mesh.y2 = np.array([0, 10, 20, 30, 40])
    mesh.zlev = np.array([0, -10, -20, -50, -100, -200, -500, -1000, -2000, -4000])
    return mesh

@pytest.fixture
def test_data():
    """Create test data for cross-section plots."""
    # Create a 2D array with depth as first dimension and x (distance/time) as second
    depth_levels = 10
    x_points = 20
    data = np.zeros((depth_levels, x_points))
    
    # Fill with some test pattern
    for d in range(depth_levels):
        for x in range(x_points):
            data[d, x] = 10 * np.sin(d/5) + 5 * np.cos(x/3) + d
    
    return data

@pytest.fixture
def xr_data():
    """Create test xarray DataArray for cross-section plots."""
    # Create a 2D array with depth as first dimension and time as second
    depth_levels = 10
    time_points = 20
    data = np.zeros((depth_levels, time_points))
    
    # Fill with some test pattern
    for d in range(depth_levels):
        for t in range(time_points):
            data[d, t] = 10 * np.sin(d/5) + 5 * np.cos(t/3) + d
    
    # Create time coordinates
    time_coords = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'D') for i in range(time_points)])
    
    # Create depth coordinates
    depth_coords = np.array([0, -10, -20, -50, -100, -200, -500, -1000, -2000, -4000])
    
    # Create xarray DataArray
    da = xr.DataArray(
        data=data,
        dims=('depth', 'time'),
        coords={
            'depth': depth_coords,
            'time': time_coords
        }
    )
    
    return da


class TestPlotTransectMap:
    """Test plot_transect_map function."""
    
    def test_plot_transect_map_no_cartopy(self):
        """Test plot_transect_map raises ImportError when cartopy is not available."""
        from pyfesom2.plotting import plot_transect_map
        
        # Create test data
        lonlat = np.vstack((np.array([0, 10, 20]), np.array([0, 10, 20])))
        mesh = mock.MagicMock()
        
        # Patch CARTOPY_AVAILABLE to False
        with mock.patch('pyfesom2.plotting.CARTOPY_AVAILABLE', False):
            with pytest.raises(ImportError, match="Cartopy is required for plotting"):
                plot_transect_map(lonlat, mesh)
    
    def test_plot_transect_map_with_mocked_cartopy(self, mock_cartopy):
        """Test plot_transect_map with mocked cartopy."""
        from pyfesom2.plotting import plot_transect_map
        
        # Mock transect_get_nodes function and other dependencies
        with mock.patch('pyfesom2.plotting.transect_get_nodes') as mock_get_nodes, \
             mock.patch('matplotlib.pyplot.subplot') as mock_subplot:
            
            # Configure mocks
            mock_get_nodes.return_value = [0, 1, 2]
            mock_ax = mock.MagicMock()
            mock_subplot.return_value = mock_ax
            
            # Create test data
            lonlat = np.vstack((np.array([0, 10, 20]), np.array([0, 10, 20])))
            mesh = mock.MagicMock()
            mesh.x2 = np.array([0, 10, 20])
            mesh.y2 = np.array([0, 10, 20])
            
            # Call function
            ax = plot_transect_map(lonlat, mesh)
            
            # Verify transect_get_nodes was called with correct arguments
            mock_get_nodes.assert_called_once_with(lonlat, mesh)
            
            # Verify ax.scatter was called for both transect points and mesh nodes
            assert ax.scatter.call_count == 2
            
            # Verify coastlines was called
            ax.coastlines.assert_called_once_with(resolution="50m")
    
    def test_plot_transect_map_projections(self, mock_cartopy):
        """Test plot_transect_map with different projections."""
        from pyfesom2.plotting import plot_transect_map
        
        # Mock transect_get_nodes function and other dependencies
        with mock.patch('pyfesom2.plotting.transect_get_nodes') as mock_get_nodes, \
             mock.patch('matplotlib.pyplot.subplot') as mock_subplot:
            
            # Configure mocks
            mock_get_nodes.return_value = [0, 1, 2]
            mock_ax = mock.MagicMock()
            mock_subplot.return_value = mock_ax
            
            # Create test data
            lonlat = np.vstack((np.array([0, 10, 20]), np.array([0, 10, 20])))
            mesh = mock.MagicMock()
            mesh.x2 = np.array([0, 10, 20])
            mesh.y2 = np.array([0, 10, 20])
            
            # Test with different projections
            for view, expected_proj in [
                ('w', 'Mercator'),
                ('np', 'NorthPolarStereo'),
                ('sp', 'SouthPolarStereo')
            ]:
                # Call function with this projection
                ax = plot_transect_map(lonlat, mesh, view=view)
                
                # For different views, we should see different projection classes used
                if view == 'w':
                    mock_cartopy.Mercator.assert_called()
                elif view == 'np':
                    mock_cartopy.NorthPolarStereo.assert_called()
                elif view == 'sp':
                    mock_cartopy.SouthPolarStereo.assert_called()
    
    def test_plot_transect_map_with_stock_img(self, mock_cartopy):
        """Test plot_transect_map with stock_img=True."""
        from pyfesom2.plotting import plot_transect_map
        
        # Mock dependencies
        with mock.patch('pyfesom2.plotting.transect_get_nodes') as mock_get_nodes, \
             mock.patch('matplotlib.pyplot.subplot') as mock_subplot:
            
            # Configure mocks
            mock_get_nodes.return_value = [0, 1, 2]
            mock_ax = mock.MagicMock()
            mock_subplot.return_value = mock_ax
            
            # Create test data
            lonlat = np.vstack((np.array([0, 10, 20]), np.array([0, 10, 20])))
            mesh = mock.MagicMock()
            mesh.x2 = np.array([0, 10, 20])
            mesh.y2 = np.array([0, 10, 20])
            
            # Call function with stock_img=True
            ax = plot_transect_map(lonlat, mesh, stock_img=True)
            
            # Verify stock_img was called
            ax.stock_img.assert_called_once()


class TestXyzPlotOne:
    """Test xyz_plot_one function."""
    
    def test_xyz_plot_one_basic(self, mock_mesh, test_data):
        """Test basic usage of xyz_plot_one."""
        from pyfesom2.plotting import xyz_plot_one
        
        # Mock dependencies
        with mock.patch('pyfesom2.plotting.ind_for_depth') as mock_ind_for_depth, \
             mock.patch('pyfesom2.plotting.get_cmap') as mock_get_cmap, \
             mock.patch('pyfesom2.plotting.get_plot_levels') as mock_get_plot_levels, \
             mock.patch('matplotlib.pyplot.gca') as mock_gca, \
             mock.patch('matplotlib.pyplot.colorbar') as mock_colorbar, \
             mock.patch('pyfesom2.plotting.sfmt', new=mock.MagicMock()):
            
            # Configure mocks
            mock_ind_for_depth.return_value = 5  # Return a depth index
            mock_get_cmap.return_value = 'viridis'  # Return a colormap
            mock_get_plot_levels.return_value = np.linspace(0, 20, 21)  # Return levels
            
            # Create mock axis
            mock_axis = mock.MagicMock()
            mock_gca.return_value = mock_axis
            
            # Create mock colorbar
            mock_cb = mock.MagicMock()
            mock_colorbar.return_value = mock_cb
            
            # Create xvals
            xvals = np.linspace(0, 10, test_data.shape[1])
            
            # Call function
            image = xyz_plot_one(
                mesh=mock_mesh,
                data=test_data,
                xvals=xvals,
                maxdepth=500,
                title="Test Plot"
            )
            
            # Verify ind_for_depth was called with correct args
            mock_ind_for_depth.assert_called_once_with(500, mock_mesh)
            
            # Verify get_cmap was called
            mock_get_cmap.assert_called_once()
            
            # Verify get_plot_levels was called with correct args
            mock_get_plot_levels.assert_called_once()
            
            # Verify contourf was called 
            mock_axis.contourf.assert_called_once()
            
            # Verify axis configuration
            mock_axis.invert_yaxis.assert_called_once()
            mock_axis.set_title.assert_called_once_with("Test Plot", size=12)
            mock_axis.set_xlabel.assert_called_once_with("Time", size=12)
            mock_axis.set_ylabel.assert_called_once_with("Depth, m", size=12)
            mock_axis.set_facecolor.assert_called_once_with("lightgray")
            
            # Verify colorbar was created
            mock_colorbar.assert_called_once()
    
    def test_xyz_plot_one_with_existing_axis(self, mock_mesh, test_data):
        """Test xyz_plot_one with an existing axis."""
        from pyfesom2.plotting import xyz_plot_one
        
        # Mock dependencies
        with mock.patch('pyfesom2.plotting.ind_for_depth') as mock_ind_for_depth, \
             mock.patch('pyfesom2.plotting.get_cmap') as mock_get_cmap, \
             mock.patch('pyfesom2.plotting.get_plot_levels') as mock_get_plot_levels, \
             mock.patch('matplotlib.pyplot.colorbar') as mock_colorbar, \
             mock.patch('pyfesom2.plotting.sfmt', new=mock.MagicMock()):
            
            # Configure mocks
            mock_ind_for_depth.return_value = 5  # Return a depth index
            mock_get_cmap.return_value = 'viridis'  # Return a colormap
            mock_get_plot_levels.return_value = np.linspace(0, 20, 21)  # Return levels
            
            # Create existing mock axis
            existing_axis = mock.MagicMock()
            
            # Create xvals
            xvals = np.linspace(0, 10, test_data.shape[1])
            
            # Call function with existing axis
            image = xyz_plot_one(
                mesh=mock_mesh,
                data=test_data,
                xvals=xvals,
                ax=existing_axis
            )
            
            # Verify existing axis was used
            existing_axis.contourf.assert_called_once()
            
            # Verify colorbar was NOT created (since we're using an existing axis)
            mock_colorbar.assert_not_called()
    
    def test_xyz_plot_one_with_custom_params(self, mock_mesh, test_data):
        """Test xyz_plot_one with custom parameters."""
        from pyfesom2.plotting import xyz_plot_one
        
        # Mock dependencies
        with mock.patch('pyfesom2.plotting.ind_for_depth') as mock_ind_for_depth, \
             mock.patch('pyfesom2.plotting.get_cmap') as mock_get_cmap, \
             mock.patch('pyfesom2.plotting.get_plot_levels') as mock_get_plot_levels, \
             mock.patch('matplotlib.pyplot.gca') as mock_gca, \
             mock.patch('matplotlib.pyplot.colorbar') as mock_colorbar, \
             mock.patch('pyfesom2.plotting.sfmt', new=mock.MagicMock()):
            
            # Configure mocks
            mock_ind_for_depth.return_value = 5  # Return a depth index
            mock_get_cmap.return_value = 'plasma'  # Return custom colormap
            mock_get_plot_levels.return_value = np.linspace(-10, 30, 21)  # Return custom levels
            
            # Create mock axis and colorbar
            mock_axis = mock.MagicMock()
            mock_gca.return_value = mock_axis
            mock_cb = mock.MagicMock()
            mock_colorbar.return_value = mock_cb
            
            # Create xvals
            xvals = np.linspace(0, 10, test_data.shape[1])
            
            # Call function with custom parameters
            image = xyz_plot_one(
                mesh=mock_mesh,
                data=test_data,
                xvals=xvals,
                levels=[-10, 30, 21],
                maxdepth=2000,
                label="Temperature (K)",
                title="Custom Title",
                cmap="plasma",
                facecolor="lightblue",
                fontsize=14,
                xlabel="Distance (km)"
            )
            
            # Verify get_cmap was called with custom cmap
            mock_get_cmap.assert_called_once_with(cmap="plasma")
            
            # Verify get_plot_levels was called with custom levels
            mock_get_plot_levels.assert_called_once()
            
            # Verify custom title, labels, and appearance
            mock_axis.set_title.assert_called_once_with("Custom Title", size=14)
            mock_axis.set_xlabel.assert_called_once_with("Distance (km)", size=14)
            mock_axis.set_ylabel.assert_called_once_with("Depth, m", size=14)
            mock_axis.set_facecolor.assert_called_once_with("lightblue")
            
            # Verify colorbar used custom label
            mock_cb.set_label.assert_called_once_with("Temperature (K)", size=14)


class TestDeprecatedFunctions:
    """Test deprecated plotting functions."""
    
    def test_plot_transect_deprecated(self):
        """Test that plot_transect raises a DeprecationWarning."""
        from pyfesom2.plotting import plot_transect
        
        with pytest.raises(DeprecationWarning, match="The plot_transect function is deprecated"):
            plot_transect()
    
    def test_hofm_plot_deprecated(self):
        """Test that hofm_plot raises a DeprecationWarning."""
        from pyfesom2.plotting import hofm_plot
        
        with pytest.raises(DeprecationWarning, match="The hovm_plot function is deprecated"):
            hofm_plot()


class TestPlotXyz:
    """Test plot_xyz function."""
    
    def test_plot_xyz_without_xvals(self, mock_mesh, test_data):
        """Test plot_xyz raises error when numpy array is provided without xvals."""
        from pyfesom2.plotting import plot_xyz
        
        # Call function with numpy array but no xvals
        with pytest.raises(ValueError, match="You provide np.array as an input, but did not provide xvals"):
            plot_xyz(
                mesh=mock_mesh,
                data=test_data  # No xvals provided
            )