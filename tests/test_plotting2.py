#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `pyfesom2` package's plotting functionality."""

import os
import pytest
import numpy as np
import xarray as xr
from pathlib import Path
from unittest import mock
from typing import List, Tuple, Union, Optional

import matplotlib.pyplot as plt

# Get the directory of the current file
THIS_DIR = Path(__file__).parent.absolute()
TEST_DATA_DIR = THIS_DIR / 'data'
PI_GRID_PATH = TEST_DATA_DIR / 'pi-grid'

# Mock cartopy modules
@pytest.fixture
def mock_cartopy():
    """Mock cartopy modules for testing."""
    with mock.patch('pyfesom2.plotting.CARTOPY_AVAILABLE', True):
        with mock.patch('pyfesom2.plotting.ccrs') as mock_ccrs:
            # Create mock projection objects with _as_mpl_axes method
            for proj in ['Mercator', 'PlateCarree', 'NorthPolarStereo', 'SouthPolarStereo', 'Robinson']:
                proj_mock = mock.MagicMock()
                # Configure _as_mpl_axes to return a tuple of (GeoAxes, {})
                proj_mock._as_mpl_axes.return_value = (mock.MagicMock(), {})
                getattr(mock_ccrs, proj).return_value = proj_mock
            yield mock_ccrs

@pytest.fixture
def fake_data():
    """Create fake data for testing plotting functions."""
    # Create a small dataset with known min/max values
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    return data

class TestCreateProjFigure:
    """Test create_proj_figure function."""

    def test_create_proj_figure_with_mocked_cartopy(self, mock_cartopy):
        """Test create_proj_figure with mocked cartopy."""
        from pyfesom2.plotting import create_proj_figure
        
        # Since we're mocking the projection functionality,
        # we'll focus on testing the function's logic rather than
        # the actual matplotlib/cartopy integration
        
        # Test that the function uses the correct projection for each string input
        create_proj_figure('merc', (1, 1), (10, 10))
        mock_cartopy.Mercator.assert_called_once()
        
        mock_cartopy.reset_mock()
        create_proj_figure('pc', (1, 1), (10, 10))
        mock_cartopy.PlateCarree.assert_called_once()
        
        mock_cartopy.reset_mock()
        create_proj_figure('np', (1, 1), (10, 10))
        mock_cartopy.NorthPolarStereo.assert_called_once()
        
        mock_cartopy.reset_mock()
        create_proj_figure('sp', (1, 1), (10, 10))
        mock_cartopy.SouthPolarStereo.assert_called_once()
        
        mock_cartopy.reset_mock()
        create_proj_figure('rob', (1, 1), (10, 10))
        mock_cartopy.Robinson.assert_called_once()
        
        # Test with invalid projection
        with pytest.raises(ValueError, match="Projection invalid is not supported"):
            create_proj_figure('invalid', (1, 1), (10, 10))
    
    def test_create_proj_figure_no_cartopy(self):
        """Test create_proj_figure when cartopy is not available."""
        from pyfesom2.plotting import create_proj_figure
        
        # Patch CARTOPY_AVAILABLE to False
        with mock.patch('pyfesom2.plotting.CARTOPY_AVAILABLE', False):
            with pytest.raises(ImportError, match="Cartopy is required for projection plots"):
                create_proj_figure('pc', (1, 1), (10, 10))


class TestGetPlotLevels:
    """Test get_plot_levels function."""
    
    def test_get_plot_levels_with_three_values(self, fake_data):
        """Test get_plot_levels with three values."""
        from pyfesom2.plotting import get_plot_levels
        
        # Test with three values (min, max, num)
        levels = [0, 10, 5]
        result = get_plot_levels(levels, fake_data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 5
        assert result[0] == 0
        assert result[-1] == 10
        
    def test_get_plot_levels_with_more_values(self, fake_data):
        """Test get_plot_levels with more than three values."""
        from pyfesom2.plotting import get_plot_levels
        
        # Test with more than three values (direct levels)
        levels = [0, 2, 4, 6, 8, 10]
        result = get_plot_levels(levels, fake_data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 6
        assert np.array_equal(result, np.array(levels))
        
    def test_get_plot_levels_with_less_values(self, fake_data):
        """Test get_plot_levels with less than three values."""
        from pyfesom2.plotting import get_plot_levels
        
        # Test with less than three values (should raise ValueError)
        with pytest.raises(ValueError, match="Levels can be the list or numpy array with three or more elements"):
            get_plot_levels([0, 10], fake_data)
            
    def test_get_plot_levels_with_none(self, fake_data):
        """Test get_plot_levels with None."""
        from pyfesom2.plotting import get_plot_levels
        
        # Test with None (should use data min/max)
        result = get_plot_levels(None, fake_data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 40  # Default number of levels
        assert result[0] == np.min(fake_data)
        assert result[-1] == np.max(fake_data)
        
    def test_get_plot_levels_with_lev_to_data(self, fake_data):
        """Test get_plot_levels with lev_to_data=True."""
        from pyfesom2.plotting import get_plot_levels, levels_to_data
        
        # Mock levels_to_data function to verify it's called
        with mock.patch('pyfesom2.plotting.levels_to_data') as mock_levels_to_data:
            mock_levels_to_data.return_value = (1.0, 9.0)  # min, max of fake_data
            
            levels = [0, 10, 5]
            result = get_plot_levels(levels, fake_data, lev_to_data=True)
            
            # Verify levels_to_data was called with correct args
            mock_levels_to_data.assert_called_once_with(0, 10, fake_data)
            
            # Verify result uses values from levels_to_data
            assert result[0] == 1.0
            assert result[-1] == 9.0


class TestLevelsToData:
    """Test levels_to_data function."""
    
    def test_levels_to_data_in_range(self, fake_data):
        """Test levels_to_data with min/max within data range."""
        from pyfesom2.plotting import levels_to_data
        
        # Test with min/max within data range
        mmin, mmax = levels_to_data(2.0, 8.0, fake_data)
        
        assert mmin == 2.0
        assert mmax == 8.0
        
    def test_levels_to_data_out_of_range(self, fake_data, capfd):
        """Test levels_to_data with min/max outside data range."""
        from pyfesom2.plotting import levels_to_data
        
        # Test with min/max outside data range
        mmin, mmax = levels_to_data(0.0, 10.0, fake_data)
        
        # Should be adjusted to data min/max
        assert mmin == 1.0
        assert mmax == 9.0
        
        # Should print warning messages
        out, _ = capfd.readouterr()
        assert "Minimum level changed" in out
        assert "Maximum level changed" in out
        
    def test_levels_to_data_with_xarray(self):
        """Test levels_to_data with xarray data."""
        from pyfesom2.plotting import levels_to_data
        
        # Create xarray DataArray
        data = xr.DataArray(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            dims=('y', 'x'),
            coords={'y': [0, 1], 'x': [0, 1, 2]}
        )
        
        # Test with min/max outside data range
        mmin, mmax = levels_to_data(0.0, 10.0, data)
        
        # Should be adjusted to data min/max
        assert mmin == 1.0
        assert mmax == 6.0


class TestInterpolateForPlot:
    """Test interpolate_for_plot function."""
    
    @pytest.fixture
    def mock_fesom2regular(self):
        """Mock the fesom2regular function."""
        with mock.patch('pyfesom2.plotting.fesom2regular') as mock_f2r:
            # Make the mock return a recognizable array for each call
            mock_f2r.side_effect = lambda *args, **kwargs: np.ones((5, 5)) * len(mock_f2r.call_args_list)
            yield mock_f2r
    
    @pytest.fixture
    def test_data(self):
        """Create test data for interpolation."""
        return [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0])
        ]
    
    @pytest.fixture
    def test_mesh(self):
        """Create a mock mesh object."""
        mesh = mock.MagicMock()
        return mesh
    
    @pytest.fixture
    def test_grid(self):
        """Create test grid coordinates."""
        lon = np.linspace(-180, 180, 10)
        lat = np.linspace(-90, 90, 10)
        lonreg2, latreg2 = np.meshgrid(lon, lat)
        return lonreg2, latreg2
    
    def test_interpolate_for_plot_nn(self, mock_fesom2regular, test_data, test_mesh, test_grid):
        """Test interpolate_for_plot with nearest neighbor interpolation."""
        from pyfesom2.plotting import interpolate_for_plot
        
        lonreg2, latreg2 = test_grid
        result = interpolate_for_plot(
            test_data, test_mesh, lonreg2, latreg2, interp="nn"
        )
        
        # Check that fesom2regular was called with the right arguments
        assert mock_fesom2regular.call_count == len(test_data)
        
        # Check first call args
        args, kwargs = mock_fesom2regular.call_args_list[0]
        assert args[0] is test_data[0]  # data
        assert args[1] is test_mesh     # mesh
        assert np.array_equal(args[2], lonreg2)  # lonreg2
        assert np.array_equal(args[3], latreg2)  # latreg2
        assert 'how' not in kwargs  # nn is default, so no 'how' specified
        
        # Check that result has the expected number of arrays
        assert len(result) == len(test_data)
    
    def test_interpolate_for_plot_idist(self, mock_fesom2regular, test_data, test_mesh, test_grid):
        """Test interpolate_for_plot with inverse distance interpolation."""
        from pyfesom2.plotting import interpolate_for_plot
        
        lonreg2, latreg2 = test_grid
        result = interpolate_for_plot(
            test_data, test_mesh, lonreg2, latreg2, interp="idist"
        )
        
        # Check fesom2regular call arguments for idist
        args, kwargs = mock_fesom2regular.call_args_list[0]
        assert kwargs.get('how') == "idist"
        assert kwargs.get('k') == 5
    
    def test_interpolate_for_plot_linear(self, mock_fesom2regular, test_data, test_mesh, test_grid):
        """Test interpolate_for_plot with linear interpolation."""
        from pyfesom2.plotting import interpolate_for_plot
        
        lonreg2, latreg2 = test_grid
        qhull_path = "/path/to/qhull"
        result = interpolate_for_plot(
            test_data, test_mesh, lonreg2, latreg2, interp="linear", qhull_path=qhull_path
        )
        
        # Check fesom2regular call arguments for linear
        args, kwargs = mock_fesom2regular.call_args_list[0]
        assert kwargs.get('how') == "linear"
        assert kwargs.get('qhull_path') == qhull_path
    
    def test_interpolate_for_plot_cubic(self, mock_fesom2regular, test_data, test_mesh, test_grid):
        """Test interpolate_for_plot with cubic interpolation."""
        from pyfesom2.plotting import interpolate_for_plot
        
        lonreg2, latreg2 = test_grid
        result = interpolate_for_plot(
            test_data, test_mesh, lonreg2, latreg2, interp="cubic"
        )
        
        # Check fesom2regular call arguments for cubic
        args, kwargs = mock_fesom2regular.call_args_list[0]
        assert kwargs.get('how') == "cubic"
    
    def test_interpolate_for_plot_invalid_method(self, test_data, test_mesh, test_grid):
        """Test interpolate_for_plot with an invalid interpolation method."""
        from pyfesom2.plotting import interpolate_for_plot
        
        lonreg2, latreg2 = test_grid
        with pytest.raises(ValueError, match="Interpolation method 'invalid' not supported"):
            interpolate_for_plot(
                test_data, test_mesh, lonreg2, latreg2, interp="invalid"
            )
    
    def test_interpolate_for_plot_with_paths(self, mock_fesom2regular, test_data, test_mesh, test_grid):
        """Test interpolate_for_plot with all path parameters specified."""
        from pyfesom2.plotting import interpolate_for_plot
        
        lonreg2, latreg2 = test_grid
        distances_path = "/path/to/distances"
        inds_path = "/path/to/inds"
        radius_of_influence = 100000
        basepath = "/base/path"
        
        result = interpolate_for_plot(
            test_data, test_mesh, lonreg2, latreg2, 
            interp="nn",
            distances_path=distances_path,
            inds_path=inds_path,
            radius_of_influence=radius_of_influence,
            basepath=basepath
        )
        
        # Check that all path parameters were passed to fesom2regular
        args, kwargs = mock_fesom2regular.call_args_list[0]
        assert kwargs.get('distances_path') == distances_path
        assert kwargs.get('inds_path') == inds_path
        assert kwargs.get('radius_of_influence') == radius_of_influence
        assert kwargs.get('basepath') == basepath

class TestPlot:
    """Test plot function."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all dependencies of the plot function."""
        with mock.patch('pyfesom2.plotting.CARTOPY_AVAILABLE', True), \
             mock.patch('pyfesom2.plotting.get_cmap') as mock_get_cmap, \
             mock.patch('pyfesom2.plotting.interpolate_for_plot') as mock_interpolate, \
             mock.patch('pyfesom2.plotting.mask_ne') as mock_mask_ne, \
             mock.patch('pyfesom2.plotting.create_proj_figure') as mock_create_fig, \
             mock.patch('pyfesom2.plotting.get_plot_levels') as mock_get_levels, \
             mock.patch('pyfesom2.plotting.add_cyclic_point') as mock_add_cyclic, \
             mock.patch('pyfesom2.plotting.ccrs') as mock_ccrs, \
             mock.patch('pyfesom2.plotting.cfeature') as mock_cfeature:
            
            # Configure return values for mocks
            mock_get_cmap.return_value = 'coolwarm'
            
            mock_mask_ne.return_value = np.zeros((10, 10), dtype=bool)
            
            # Create mock figure and axes
            mock_fig = mock.MagicMock()
            mock_ax = mock.MagicMock()
            mock_create_fig.return_value = (mock_fig, mock_ax)
            
            # Mock data levels
            mock_get_levels.return_value = np.linspace(0, 10, 11)
            
            # Mock cyclic point
            mock_add_cyclic.return_value = (np.ones((10, 11)), np.linspace(0, 360, 11))
            
            yield {
                'get_cmap': mock_get_cmap,
                'interpolate': mock_interpolate,
                'mask_ne': mock_mask_ne,
                'create_fig': mock_create_fig,
                'get_levels': mock_get_levels,
                'add_cyclic': mock_add_cyclic,
                'ccrs': mock_ccrs,
                'cfeature': mock_cfeature,
                'fig': mock_fig,
                'ax': mock_ax
            }
    
    @pytest.fixture
    def test_data(self):
        """Create test data."""
        return np.ones((100,))
    
    @pytest.fixture
    def test_mesh(self):
        """Create a mock mesh object."""
        return mock.MagicMock()
    
    def test_plot_no_cartopy(self, test_data, test_mesh):
        """Test plot function raises ImportError when cartopy is not available."""
        from pyfesom2.plotting import plot
        
        with mock.patch('pyfesom2.plotting.CARTOPY_AVAILABLE', False):
            with pytest.raises(ImportError, match="Cartopy is required for plotting"):
                plot(test_mesh, test_data)
    
    def test_plot_data_conversion(self, mock_dependencies, test_data, test_mesh):
        """Test plot function converts single data array to list."""
        from pyfesom2.plotting import plot
        
        mock_dependencies['interpolate'].return_value = [np.ones((10, 10))]
        
        plot(test_mesh, test_data)
        
        # Check that data was converted to list
        args, _ = mock_dependencies['interpolate'].call_args
        assert isinstance(args[0], list)
        assert args[0][0] is test_data
    
    def test_plot_title_validation(self, mock_dependencies, test_mesh):
        """Test plot function validates titles."""
        from pyfesom2.plotting import plot
        
        data = [np.ones((100,)), np.ones((100,))]
        titles = "Single Title"
        
        with pytest.raises(ValueError, match="number of titles do not match"):
            plot(test_mesh, data, titles=titles)
    
    def test_plot_rowscol_validation(self, mock_dependencies, test_mesh):
        """Test plot function validates rows*columns."""
        from pyfesom2.plotting import plot
        
        data = [np.ones((100,)), np.ones((100,)), np.ones((100,))]
        rowscol = (1, 1)
        
        with pytest.raises(ValueError, match="Number of rows.*columns is smaller than"):
            plot(test_mesh, data, rowscol=rowscol)
    
    def test_plot_without_interpolation(self, mock_dependencies, test_mesh):
        """Test plot function when interpolated_data is provided."""
        from pyfesom2.plotting import plot
        
        # Provide pre-interpolated data
        interpolated_data = np.ones((10, 10))
        lonreg = np.linspace(-180, 180, 10)
        latreg = np.linspace(-90, 90, 10)
        
        # Call plot function
        plot(
            test_mesh, 
            data=np.ones((100,)),  # This should be ignored
            interpolated_data=interpolated_data,
            lonreg=lonreg,
            latreg=latreg
        )
        
        # Verify interpolate_for_plot was not called
        mock_dependencies['interpolate'].assert_not_called()
    
    def test_plot_with_interpolation(self, mock_dependencies, test_data, test_mesh):
        """Test plot function performs interpolation when needed."""
        from pyfesom2.plotting import plot
        
        # Configure mock
        mock_dependencies['interpolate'].return_value = [np.ones((10, 10))]
        
        # Call plot function
        plot(test_mesh, test_data)
        
        # Verify interpolate_for_plot was called
        mock_dependencies['interpolate'].assert_called_once()
    
    def test_plot_no_pi_mask_true(self, mock_dependencies, test_data, test_mesh):
        """Test plot function with no_pi_mask=True."""
        from pyfesom2.plotting import plot
        import numpy.ma as ma
        
        # Mock interpolation result
        interpolated = [np.ones((10, 10))]
        mock_dependencies['interpolate'].return_value = interpolated
        
        # Call plot function with no_pi_mask=True
        plot(test_mesh, test_data, no_pi_mask=True)
        
        # Check masked_where was not called (we're using a mock)
        # but we can verify mask_ne was called
        mock_dependencies['mask_ne'].assert_called_once()
    
    def test_plot_type_cf(self, mock_dependencies, test_data, test_mesh):
        """Test plot function with ptype='cf' (contourf)."""
        from pyfesom2.plotting import plot
        
        # Mock interpolation result
        mock_dependencies['interpolate'].return_value = [np.ones((10, 10))]
        
        # Call plot function with ptype='cf'
        plot(test_mesh, test_data, ptype='cf')
        
        # Check that contourf was called on ax
        mock_ax = mock_dependencies['ax']
        if isinstance(mock_ax, list):
            mock_ax = mock_ax[0]
        mock_ax.contourf.assert_called_once()
        mock_ax.pcolormesh.assert_not_called()
    
    def test_plot_type_pcm(self, mock_dependencies, test_data, test_mesh):
        """Test plot function with ptype='pcm' (pcolormesh)."""
        from pyfesom2.plotting import plot
        
        # Mock interpolation result
        mock_dependencies['interpolate'].return_value = [np.ones((10, 10))]
        
        # Call plot function with ptype='pcm'
        plot(test_mesh, test_data, ptype='pcm')
        
        # Check that pcolormesh was called on ax
        mock_ax = mock_dependencies['ax']
        if isinstance(mock_ax, list):
            mock_ax = mock_ax[0]
        mock_ax.pcolormesh.assert_called_once()
        mock_ax.contourf.assert_not_called()
    
    def test_plot_invalid_type(self, mock_dependencies, test_data, test_mesh):
        """Test plot function with invalid ptype."""
        from pyfesom2.plotting import plot
        
        # Mock interpolation result
        mock_dependencies['interpolate'].return_value = [np.ones((10, 10))]
        
        # Call plot function with invalid ptype
        with pytest.raises(ValueError, match="Unknown plot type"):
            plot(test_mesh, test_data, ptype='invalid')
    
    def test_plot_with_units(self, mock_dependencies, test_data, test_mesh):
        """Test plot function with units."""
        from pyfesom2.plotting import plot
        
        # Mock interpolation result
        mock_dependencies['interpolate'].return_value = [np.ones((10, 10))]
        
        # Mock colorbar
        mock_colorbar = mock.MagicMock()
        mock_dependencies['fig'].colorbar.return_value = mock_colorbar
        
        # Call plot function with units
        units = "Temperature (°C)"
        plot(test_mesh, test_data, units=units)
        
        # Check that colorbar label was set
        mock_colorbar.set_label.assert_called_once_with(units, size=20)

class TestPlotVector:
    """Test plot_vector function."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all dependencies of the plot_vector function."""
        with mock.patch('pyfesom2.plotting.CARTOPY_AVAILABLE', True), \
             mock.patch('pyfesom2.plotting.get_cmap') as mock_get_cmap, \
             mock.patch('pyfesom2.plotting.create_proj_figure') as mock_create_fig, \
             mock.patch('pyfesom2.plotting.ccrs') as mock_ccrs, \
             mock.patch('matplotlib.colors.Normalize') as mock_norm:
            
            # Configure return values for mocks
            mock_get_cmap.return_value = 'coolwarm'
            
            # Create mock figure and axes
            mock_fig = mock.MagicMock()
            mock_ax = mock.MagicMock()
            mock_create_fig.return_value = (mock_fig, mock_ax)
            
            # Mock normalization
            mock_norm_instance = mock.MagicMock()
            mock_norm.return_value = mock_norm_instance
            
            yield {
                'get_cmap': mock_get_cmap,
                'create_fig': mock_create_fig,
                'ccrs': mock_ccrs,
                'norm': mock_norm,
                'norm_instance': mock_norm_instance,
                'fig': mock_fig,
                'ax': mock_ax
            }
    
    @pytest.fixture
    def test_vector_data(self):
        """Create test vector data."""
        # Create small u and v components for testing
        u_int = np.ones((10, 10))
        v_int = np.ones((10, 10)) * 0.5
        
        # Create grid
        lon = np.linspace(-180, 180, 10)
        lat = np.linspace(-90, 90, 10)
        lonreg2, latreg2 = np.meshgrid(lon, lat)
        
        return u_int, v_int, lonreg2, latreg2
    
    def test_plot_vector_no_cartopy(self, test_vector_data):
        """Test plot_vector function raises ImportError when cartopy is not available."""
        from pyfesom2.plotting import plot_vector
        
        u_int, v_int, lonreg2, latreg2 = test_vector_data
        
        with mock.patch('pyfesom2.plotting.CARTOPY_AVAILABLE', False):
            with pytest.raises(ImportError, match="Cartopy is required for plotting"):
                plot_vector(u_int, v_int, lonreg2, latreg2)
    
    def test_plot_vector_data_conversion(self, mock_dependencies, test_vector_data):
        """Test plot_vector function converts single arrays to lists."""
        from pyfesom2.plotting import plot_vector
        
        u_int, v_int, lonreg2, latreg2 = test_vector_data
        
        plot_vector(u_int, v_int, lonreg2, latreg2)
        
        # The function should have converted u_int and v_int to lists
        mock_ax = mock_dependencies['ax']
        if isinstance(mock_ax, list):
            mock_ax = mock_ax[0]
        
        # Check that quiver was called once
        mock_ax.quiver.assert_called_once()
    
    def test_plot_vector_title_validation(self, mock_dependencies, test_vector_data):
        """Test plot_vector function validates titles."""
        from pyfesom2.plotting import plot_vector
        
        u_int, v_int, lonreg2, latreg2 = test_vector_data
        
        # Create lists of u and v components
        u_list = [u_int, u_int]
        v_list = [v_int, v_int]
        
        # Create a single title (not matching the number of vector fields)
        titles = "Single Title"
        
        with pytest.raises(ValueError, match="number of titles do not match"):
            plot_vector(u_list, v_list, lonreg2, latreg2, titles=titles)
    
    def test_plot_vector_rowscol_validation(self, mock_dependencies, test_vector_data):
        """Test plot_vector function validates rows*columns."""
        from pyfesom2.plotting import plot_vector
        
        u_int, v_int, lonreg2, latreg2 = test_vector_data
        
        # Create lists of u and v components (3 fields)
        u_list = [u_int, u_int, u_int]
        v_list = [v_int, v_int, v_int]
        
        # Set rowscol to only have space for one plot
        rowscol = [1, 1]
        
        with pytest.raises(ValueError, match="Number of rows.*columns is smaller than"):
            plot_vector(u_list, v_list, lonreg2, latreg2, rowscol=rowscol)
    
    def test_plot_vector_with_specific_cmap(self, mock_dependencies, test_vector_data):
        """Test plot_vector function with specific colormap."""
        from pyfesom2.plotting import plot_vector
        
        u_int, v_int, lonreg2, latreg2 = test_vector_data
        
        # Call with specific colormap
        cmap = "viridis"
        plot_vector(u_int, v_int, lonreg2, latreg2, cmap=cmap)
        
        # Check that get_cmap was called with the specified colormap
        mock_dependencies['get_cmap'].assert_called_once_with(cmap=cmap)
    
    def test_plot_vector_with_normalization(self, mock_dependencies, test_vector_data):
        """Test plot_vector function with specific vmin/vmax."""
        from pyfesom2.plotting import plot_vector
        
        u_int, v_int, lonreg2, latreg2 = test_vector_data
        
        # Call with specific vmin/vmax
        vmin = 0.1
        vmax = 1.5
        plot_vector(u_int, v_int, lonreg2, latreg2, vmin=vmin, vmax=vmax)
        
        # Check that mpl.colors.Normalize was called with correct values
        mock_dependencies['norm'].assert_called_once_with(vmin=vmin, vmax=vmax, clip=False)
    
    def test_plot_vector_with_specific_projection(self, mock_dependencies, test_vector_data):
        """Test plot_vector function with specific map projection."""
        from pyfesom2.plotting import plot_vector
        
        u_int, v_int, lonreg2, latreg2 = test_vector_data
        
        # Call with specific map projection
        mapproj = "np"  # North Polar Stereo
        plot_vector(u_int, v_int, lonreg2, latreg2, mapproj=mapproj)
        
        # Check that create_proj_figure was called with the specified projection
        args, kwargs = mock_dependencies['create_fig'].call_args
        assert args[0] == mapproj
    
    def test_plot_vector_with_titles(self, mock_dependencies, test_vector_data):
        """Test plot_vector function with titles."""
        from pyfesom2.plotting import plot_vector
        
        u_int, v_int, lonreg2, latreg2 = test_vector_data
        
        # Call with title
        title = "Vector Field Plot"
        plot_vector(u_int, v_int, lonreg2, latreg2, titles=title)
        
        # Check that title was set on axis
        mock_ax = mock_dependencies['ax']
        if isinstance(mock_ax, list):
            mock_ax = mock_ax[0]
        mock_ax.set_title.assert_called_once_with(title, size=20)
    
    def test_plot_vector_with_units(self, mock_dependencies, test_vector_data):
        """Test plot_vector function with units."""
        from pyfesom2.plotting import plot_vector
        
        u_int, v_int, lonreg2, latreg2 = test_vector_data
        
        # Mock colorbar
        mock_colorbar = mock.MagicMock()
        mock_dependencies['fig'].colorbar.return_value = mock_colorbar
        
        # Call with specific units
        units = "Velocity (m/s)"
        plot_vector(u_int, v_int, lonreg2, latreg2, units=units)
        
        # Check that colorbar label was set correctly
        mock_colorbar.set_label.assert_called_once_with(units, size=20)
    
    def test_plot_vector_stride_and_scale(self, mock_dependencies, test_vector_data):
        """Test plot_vector function with specific stride and scale."""
        from pyfesom2.plotting import plot_vector
        
        u_int, v_int, lonreg2, latreg2 = test_vector_data
        
        # Call with specific stride and scale
        sstep = 2
        scale = 15
        plot_vector(u_int, v_int, lonreg2, latreg2, sstep=sstep, scale=scale)
        
        # Check that quiver was called with the right stride and scale
        mock_ax = mock_dependencies['ax']
        if isinstance(mock_ax, list):
            mock_ax = mock_ax[0]
        
        args, kwargs = mock_ax.quiver.call_args
        assert kwargs['scale'] == scale
        
        # Check that the first two args are strided correctly
        assert args[0].shape == lonreg2[::sstep, ::sstep].shape
        assert args[1].shape == latreg2[::sstep, ::sstep].shape
    
    def test_plot_vector_with_regrid_shape(self, mock_dependencies, test_vector_data):
        """Test plot_vector function with specific regrid_shape."""
        from pyfesom2.plotting import plot_vector
        
        u_int, v_int, lonreg2, latreg2 = test_vector_data
        
        # Call with specific regrid_shape
        regrid_shape = 50
        plot_vector(u_int, v_int, lonreg2, latreg2, regrid_shape=regrid_shape)
        
        # Check that quiver was called with the right regrid_shape
        mock_ax = mock_dependencies['ax']
        if isinstance(mock_ax, list):
            mock_ax = mock_ax[0]
        
        args, kwargs = mock_ax.quiver.call_args
        assert kwargs['regrid_shape'] == regrid_shape
    
    def test_plot_vector_with_custom_box(self, mock_dependencies, test_vector_data):
        """Test plot_vector function with custom box boundaries."""
        from pyfesom2.plotting import plot_vector
        
        u_int, v_int, lonreg2, latreg2 = test_vector_data
        
        # Call with custom box boundaries
        box = [-90, 90, -45, 45]
        plot_vector(u_int, v_int, lonreg2, latreg2, box=box)
        
        # Check that set_extent was called with the correct boundaries
        mock_ax = mock_dependencies['ax']
        if isinstance(mock_ax, list):
            mock_ax = mock_ax[0]
        
        args, kwargs = mock_ax.set_extent.call_args
        assert args[0] == box

class TestTplot:
    """Test tplot function."""
    
    @pytest.fixture
    def test_mesh(self):
        """Create a mock mesh object with required attributes."""
        mesh = mock.MagicMock()
        mesh.n2d = 100  # Number of 2D nodes
        mesh.e2d = 180  # Number of 2D elements
        mesh.x2 = np.linspace(-180, 180, 100)  # x coordinates of nodes
        mesh.y2 = np.linspace(-80, 80, 100)    # y coordinates of nodes
        return mesh
    
    @pytest.fixture
    def test_data(self):
        """Create test data for tplot."""
        return np.linspace(0, 10, 100)  # One value per node
    
    def test_tplot_no_cartopy(self, test_mesh, test_data):
        """Test tplot function raises ImportError when cartopy is not available."""
        from pyfesom2.plotting import tplot
        
        with mock.patch('pyfesom2.plotting.CARTOPY_AVAILABLE', False):
            with pytest.raises(ImportError, match="Cartopy is required for plotting"):
                tplot(test_mesh, test_data)
    
    def test_tplot_title_validation(self, test_mesh):
        """Test tplot function validates titles."""
        from pyfesom2.plotting import tplot
        
        # Mock dependencies to avoid actual execution
        with mock.patch('pyfesom2.plotting.CARTOPY_AVAILABLE', True), \
             mock.patch('pyfesom2.plotting.get_cmap'), \
             mock.patch('pyfesom2.plotting.get_plot_levels'), \
             mock.patch('pyfesom2.plotting.create_proj_figure', return_value=(mock.MagicMock(), mock.MagicMock())), \
             mock.patch('pyfesom2.plotting.cut_region'), \
             mock.patch('pyfesom2.plotting.get_no_cyclic'):
            
            data = [np.ones(100), np.ones(100)]
            titles = "Single Title"
            
            with pytest.raises(ValueError, match="number of titles do not match"):
                tplot(test_mesh, data, titles=titles)
    
    def test_tplot_rowscol_validation(self, test_mesh):
        """Test tplot function validates rows*columns."""
        from pyfesom2.plotting import tplot
        
        # Mock dependencies to avoid actual execution
        with mock.patch('pyfesom2.plotting.CARTOPY_AVAILABLE', True), \
             mock.patch('pyfesom2.plotting.get_cmap'), \
             mock.patch('pyfesom2.plotting.get_plot_levels'), \
             mock.patch('pyfesom2.plotting.create_proj_figure', return_value=(mock.MagicMock(), mock.MagicMock())), \
             mock.patch('pyfesom2.plotting.cut_region'), \
             mock.patch('pyfesom2.plotting.get_no_cyclic'):
            
            data = [np.ones(100), np.ones(100), np.ones(100)]
            rowscol = (1, 1)
            
            with pytest.raises(ValueError, match="Number of rows.*columns is smaller than"):
                tplot(test_mesh, data, rowscol=rowscol)
    
    def test_tplot_with_specific_cmap(self, test_mesh, test_data):
        """Test tplot function with specific colormap."""
        from pyfesom2.plotting import tplot
        
        # Mock dependencies and capture get_cmap call
        with mock.patch('pyfesom2.plotting.CARTOPY_AVAILABLE', True), \
             mock.patch('pyfesom2.plotting.get_cmap') as mock_get_cmap, \
             mock.patch('pyfesom2.plotting.get_plot_levels'), \
             mock.patch('pyfesom2.plotting.create_proj_figure') as mock_create_fig, \
             mock.patch('pyfesom2.plotting.cut_region', return_value=(np.array(range(180)), np.ones(180, dtype=bool))), \
             mock.patch('pyfesom2.plotting.get_no_cyclic', return_value=np.ones(180, dtype=bool)), \
             mock.patch('pyfesom2.plotting.ccrs'), \
             mock.patch('matplotlib.pyplot.Figure.colorbar', return_value=mock.MagicMock()):
            
            # Set up proper mocks for axes
            mock_fig = mock.MagicMock()
            mock_ax = mock.MagicMock()
            mock_ax.set_extent = mock.MagicMock()
            mock_ax.__getitem__ = mock.MagicMock(return_value=mock_ax)
            mock_create_fig.return_value = (mock_fig, mock_ax)
            
            # Call with specific colormap
            cmap = "viridis"
            tplot(test_mesh, test_data, cmap=cmap)
            
            # Check that get_cmap was called with the specified colormap
            mock_get_cmap.assert_called_once_with(cmap=cmap)
    
    def test_tplot_with_specific_levels(self, test_mesh, test_data):
        """Test tplot function with specific levels."""
        from pyfesom2.plotting import tplot
        
        # Mock dependencies and capture get_plot_levels call
        with mock.patch('pyfesom2.plotting.CARTOPY_AVAILABLE', True), \
             mock.patch('pyfesom2.plotting.get_cmap'), \
             mock.patch('pyfesom2.plotting.get_plot_levels') as mock_get_levels, \
             mock.patch('pyfesom2.plotting.create_proj_figure') as mock_create_fig, \
             mock.patch('pyfesom2.plotting.cut_region', return_value=(np.array(range(180)), np.ones(180, dtype=bool))), \
             mock.patch('pyfesom2.plotting.get_no_cyclic', return_value=np.ones(180, dtype=bool)), \
             mock.patch('pyfesom2.plotting.ccrs'), \
             mock.patch('matplotlib.pyplot.Figure.colorbar', return_value=mock.MagicMock()):
            
            # Set up proper mocks for axes
            mock_fig = mock.MagicMock()
            mock_ax = mock.MagicMock()
            mock_ax.set_extent = mock.MagicMock()
            mock_ax.__getitem__ = mock.MagicMock(return_value=mock_ax)
            mock_create_fig.return_value = (mock_fig, mock_ax)
            
            # Call with specific levels
            levels = [0, 10, 5]
            tplot(test_mesh, test_data, levels=levels)
            
            # Check that get_plot_levels was called with the specified levels
            mock_get_levels.assert_called_once()
            args, _ = mock_get_levels.call_args
            assert args[0] == levels
    
    def test_tplot_with_specific_projection(self, test_mesh, test_data):
        """Test tplot function with specific map projection."""
        from pyfesom2.plotting import tplot
        
        # Mock dependencies and capture create_proj_figure call
        with mock.patch('pyfesom2.plotting.CARTOPY_AVAILABLE', True), \
             mock.patch('pyfesom2.plotting.get_cmap'), \
             mock.patch('pyfesom2.plotting.get_plot_levels'), \
             mock.patch('pyfesom2.plotting.create_proj_figure') as mock_create_fig, \
             mock.patch('pyfesom2.plotting.cut_region', return_value=(np.array(range(180)), np.ones(180, dtype=bool))), \
             mock.patch('pyfesom2.plotting.get_no_cyclic', return_value=np.ones(180, dtype=bool)), \
             mock.patch('pyfesom2.plotting.ccrs'):
            
            # Set up proper mocks for axes
            mock_fig = mock.MagicMock()
            mock_ax = mock.MagicMock()
            mock_ax.set_extent = mock.MagicMock()
            mock_ax.__getitem__ = mock.MagicMock(return_value=mock_ax)
            mock_create_fig.return_value = (mock_fig, mock_ax)
            
            # Call with specific map projection
            mapproj = "np"  # North Polar Stereo
            tplot(test_mesh, test_data, mapproj=mapproj)
            
            # Check that create_proj_figure was called with the specified projection
            mock_create_fig.assert_called_once()
            args, _ = mock_create_fig.call_args
            assert args[0] == mapproj
    
    def test_tplot_with_box(self, test_mesh, test_data):
        """Test tplot function with custom box."""
        from pyfesom2.plotting import tplot
        
        # Mock dependencies and capture cut_region call
        with mock.patch('pyfesom2.plotting.CARTOPY_AVAILABLE', True), \
             mock.patch('pyfesom2.plotting.get_cmap'), \
             mock.patch('pyfesom2.plotting.get_plot_levels'), \
             mock.patch('pyfesom2.plotting.create_proj_figure') as mock_create_fig, \
             mock.patch('pyfesom2.plotting.cut_region') as mock_cut_region, \
             mock.patch('pyfesom2.plotting.get_no_cyclic'), \
             mock.patch('pyfesom2.plotting.ccrs'):
            
            # Configure mocks
            mock_fig = mock.MagicMock()
            mock_ax = mock.MagicMock()
            mock_ax.set_extent = mock.MagicMock()
            mock_ax.__getitem__ = mock.MagicMock(return_value=mock_ax)
            mock_create_fig.return_value = (mock_fig, mock_ax)
            
            elem_no_nan = np.array(range(180))
            no_nan_triangles = np.ones(180, dtype=bool)
            mock_cut_region.return_value = (elem_no_nan, no_nan_triangles)
            
            # Call with custom box
            box = [-90, 90, -45, 45]
            box_expand = 2
            tplot(test_mesh, test_data, box=box, box_expand=box_expand)
            
            # Check that cut_region was called with expanded box
            expected_box = [box[0] - box_expand, box[1] + box_expand, 
                            box[2] - box_expand, box[3] + box_expand]
            
            mock_cut_region.assert_called_once()
            args, _ = mock_cut_region.call_args
            assert args[1] == expected_box
    
    def test_tplot_data_on_elements_cf_error(self, test_mesh):
        """Test tplot raises error with data on elements using contourf."""
        from pyfesom2.plotting import tplot
        
        # Mock dependencies to avoid actual execution
        with mock.patch('pyfesom2.plotting.CARTOPY_AVAILABLE', True), \
             mock.patch('pyfesom2.plotting.get_cmap'), \
             mock.patch('pyfesom2.plotting.get_plot_levels'), \
             mock.patch('pyfesom2.plotting.create_proj_figure', return_value=(mock.MagicMock(), mock.MagicMock())), \
             mock.patch('pyfesom2.plotting.cut_region', return_value=(np.array([]), np.array([]))), \
             mock.patch('pyfesom2.plotting.get_no_cyclic'), \
             mock.patch('pyfesom2.plotting.ccrs'):
            
            # Create data sized for elements
            elem_data = np.linspace(0, 10, test_mesh.e2d)
            
            # Data on elements with contourf should raise error
            with pytest.raises(ValueError) as excinfo:
                tplot(test_mesh, elem_data, ptype="cf")
            
            # Check that the exception message contains our expected text
            assert "You are trying to plot data on elements using countourf" in str(excinfo.value)
    
    def test_tplot_invalid_data_size(self, test_mesh):
        """Test tplot raises error with invalid data size."""
        from pyfesom2.plotting import tplot
        
        # Mock dependencies to avoid actual execution
        with mock.patch('pyfesom2.plotting.CARTOPY_AVAILABLE', True), \
             mock.patch('pyfesom2.plotting.get_cmap'), \
             mock.patch('pyfesom2.plotting.get_plot_levels'), \
             mock.patch('pyfesom2.plotting.create_proj_figure', return_value=(mock.MagicMock(), mock.MagicMock())), \
             mock.patch('pyfesom2.plotting.cut_region', return_value=(np.array([]), np.array([]))), \
             mock.patch('pyfesom2.plotting.get_no_cyclic'), \
             mock.patch('pyfesom2.plotting.ccrs'):
            
            # Create data with invalid size
            bad_data = np.linspace(0, 10, 50)  # Neither nodes nor elements
            
            # Should raise error
            with pytest.raises(ValueError) as excinfo:
                tplot(test_mesh, bad_data)
            
            # Check that the exception message contains the expected text
            assert "Data size" in str(excinfo.value) and "doesn't match either nodes" in str(excinfo.value)
    
    def test_tplot_invalid_plot_type(self, test_mesh, test_data):
        """Test tplot raises error with invalid plot type."""
        from pyfesom2.plotting import tplot
        
        # Mock dependencies to avoid actual execution but let it get to the ptype check
        with mock.patch('pyfesom2.plotting.CARTOPY_AVAILABLE', True), \
             mock.patch('pyfesom2.plotting.get_cmap'), \
             mock.patch('pyfesom2.plotting.get_plot_levels'), \
             mock.patch('pyfesom2.plotting.create_proj_figure') as mock_create_fig, \
             mock.patch('pyfesom2.plotting.cut_region', return_value=(np.array(range(180)), np.ones(180, dtype=bool))), \
             mock.patch('pyfesom2.plotting.get_no_cyclic', return_value=np.ones(180, dtype=bool)), \
             mock.patch('pyfesom2.plotting.ccrs'):
            
            # Set up proper mocks for axes
            mock_fig = mock.MagicMock()
            mock_ax = mock.MagicMock()
            mock_ax.set_extent = mock.MagicMock()
            mock_ax.__getitem__ = mock.MagicMock(return_value=mock_ax)
            mock_create_fig.return_value = (mock_fig, mock_ax)
            
            # Should raise error
            with pytest.raises(ValueError) as excinfo:
                tplot(test_mesh, test_data, ptype="invalid")
            
            # Check that the exception message contains the expected text
            assert "Only `cf`" in str(excinfo.value) and "options are supported" in str(excinfo.value)
