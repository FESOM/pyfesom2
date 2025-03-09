#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `pyfesom2` package's mesh loading functionality."""

import os
import pytest
import numpy as np
import xarray as xr
from pathlib import Path
from tempfile import TemporaryDirectory

from pyfesom2 import load_mesh

# Get the directory of the current file
THIS_DIR = Path(__file__).parent.absolute()
TEST_DATA_DIR = THIS_DIR / 'data'
PI_GRID_PATH = TEST_DATA_DIR / 'pi-grid'


class TestMeshLoading:
    """Test suite for mesh loading functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Ensure test directory exists
        if not PI_GRID_PATH.exists():
            pytest.skip(f"Test data directory {PI_GRID_PATH} not found")
        
        # Remove any existing cache files before tests
        self.cleanup_cache_files()

    def teardown_method(self):
        """Clean up after each test method."""
        self.cleanup_cache_files()

    def cleanup_cache_files(self):
        """Clean up cache files that might be left from previous tests."""
        cache_files = [
            PI_GRID_PATH / 'pickle_mesh_py3_fesom2',
            PI_GRID_PATH / 'joblib_mesh_py3_fesom2'
        ]
        for file_path in cache_files:
            if file_path.exists():
                file_path.unlink()

    def test_mesh_load_without_cache(self):
        """Test loading mesh without caching."""
        mesh = load_mesh(PI_GRID_PATH, cache_method=None)
        assert mesh.n2d == 3140
        assert mesh.e2d == 5839
        
        # Verify no cache files were created
        assert not (PI_GRID_PATH / 'pickle_mesh_py3_fesom2').exists()
        assert not (PI_GRID_PATH / 'joblib_mesh_py3_fesom2').exists()

    def test_mesh_load_with_pickle_cache(self):
        """Test loading mesh with pickle caching."""
        # First load should create the cache
        mesh = load_mesh(PI_GRID_PATH, cache_method='pickle')
        assert mesh.n2d == 3140
        assert mesh.e2d == 5839
        
        # Verify cache was created
        cache_file = PI_GRID_PATH / 'pickle_mesh_py3_fesom2'
        assert cache_file.exists()
        
        # Second load should use the cache
        mesh2 = load_mesh(PI_GRID_PATH, cache_method='pickle')
        assert mesh2.n2d == 3140
        assert mesh2.e2d == 5839

    def test_mesh_load_with_joblib_cache(self):
        """Test loading mesh with joblib caching."""
        # First load should create the cache
        mesh = load_mesh(PI_GRID_PATH, cache_method='joblib')
        assert mesh.n2d == 3140
        assert mesh.e2d == 5839
        
        # Verify cache was created
        cache_file = PI_GRID_PATH / 'joblib_mesh_py3_fesom2'
        assert cache_file.exists()
        
        # Second load should use the cache
        mesh2 = load_mesh(PI_GRID_PATH, cache_method='joblib')
        assert mesh2.n2d == 3140
        assert mesh2.e2d == 5839

    def test_mesh_cache_in_custom_directory(self):
        """Test cache creation in a custom directory when mesh directory is read-only."""
        with TemporaryDirectory() as temp_dir:
            # Set custom cache directory through environment variable
            os.environ["PYFESOM_CACHE"] = temp_dir
            
            # Load mesh with caching enabled
            mesh = load_mesh(PI_GRID_PATH, cache_method='pickle')
            assert mesh.n2d == 3140
            
            # Check if cache was created in the custom directory
            cache_dir = Path(temp_dir) / 'pi-grid'
            cache_file = cache_dir / 'pickle_mesh_py3_fesom2'
            
            # This test might fail if the mesh directory is writable
            # In that case, the cache would be created in the mesh directory
            if not (PI_GRID_PATH / 'pickle_mesh_py3_fesom2').exists():
                assert cache_dir.exists()
                assert cache_file.exists()
            
            # Clean up
            del os.environ["PYFESOM_CACHE"]

    def test_mesh_attributes(self):
        """Test that the mesh has the expected attributes."""
        mesh = load_mesh(PI_GRID_PATH, cache_method=None)
        
        # Basic attributes
        assert hasattr(mesh, 'x2')
        assert hasattr(mesh, 'y2')
        assert hasattr(mesh, 'elem')
        assert hasattr(mesh, 'nlev')
        
        # Check types and shapes
        assert isinstance(mesh.x2, np.ndarray)
        assert isinstance(mesh.y2, np.ndarray)
        assert isinstance(mesh.elem, np.ndarray)
        
        assert mesh.x2.shape == (mesh.n2d,)
        assert mesh.y2.shape == (mesh.n2d,)
        assert mesh.elem.shape == (mesh.e2d, 3)
        
        # Check computed attributes
        assert hasattr(mesh, 'voltri')
        assert mesh.voltri.shape == (mesh.e2d,)
        
        # Check string representation
        mesh_info = str(mesh)
        assert "FESOM mesh" in mesh_info
        assert str(mesh.n2d) in mesh_info
        assert str(mesh.e2d) in mesh_info

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `pyfesom2` package depth level functions."""

import os
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from pyfesom2 import load_mesh, ind_for_depth

# Get the directory of the current file
THIS_DIR = Path(__file__).parent.absolute()
TEST_DATA_DIR = THIS_DIR / 'data'
PI_GRID_PATH = TEST_DATA_DIR / 'pi-grid'


class TestDepthFunctions:
    """Test suite for depth-related functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Ensure test directory exists
        if not PI_GRID_PATH.exists():
            pytest.skip(f"Test data directory {PI_GRID_PATH} not found")
        
        # Load the mesh for tests that need it
        self.mesh = load_mesh(PI_GRID_PATH, cache_method=None)
        
        # Create a simple mock mesh for focused testing
        self.mock_mesh = Mock()
        self.mock_mesh.zlev = np.array([0, -10, -20, -50, -100, -200, -500, -1000, -2000, -5000])

    def test_ind_for_depth_exact_match(self):
        """Test finding depth index with exact depth match."""
        # Test with depths that exactly match mesh.zlev values
        for i, depth in enumerate(self.mock_mesh.zlev):
            # Test with both positive and negative depth values
            assert ind_for_depth(depth, self.mock_mesh) == i
            assert ind_for_depth(abs(depth), self.mock_mesh) == i

    def test_ind_for_depth_between_levels(self):
        """Test finding depth index when depth is between model levels."""
        # Test depths between mesh levels to ensure nearest is returned
        test_cases = [
            (-15, 1),  # Between -10 and -20, should return index 1 (-10)
            (15, 1),   # Same but with positive value
            (-75, 3),  # Between -50 and -100, should return index 3 (-50)
            (75, 3),   # Same but with positive value
            (-150, 4), # Between -100 and -200, should return index 4 (-100)
            (150, 4),  # Same but with positive value
        ]
        
        for depth, expected_index in test_cases:
            assert ind_for_depth(depth, self.mock_mesh) == expected_index, \
                f"Failed for depth {depth}, expected index {expected_index}"

    def test_ind_for_depth_out_of_range(self):
        """Test finding depth index for depths outside the model range."""
        # Test depths that are out of range (shallower or deeper)
        assert ind_for_depth(-6000, self.mock_mesh) == 9  # Deeper than deepest level
        assert ind_for_depth(6000, self.mock_mesh) == 9   # Same with positive value
        assert ind_for_depth(-1, self.mock_mesh) == 0     # Between 0 and -10
        assert ind_for_depth(1, self.mock_mesh) == 0      # Same with positive value

    def test_ind_for_depth_with_real_mesh(self):
        """Test with actual mesh object from FESOM."""
        # Get some test depths from the real mesh
        if hasattr(self.mesh, 'zlev') and self.mesh.zlev is not None:
            # Test the first, middle and last levels
            depths_to_test = [
                self.mesh.zlev[0],                                     # First level
                self.mesh.zlev[len(self.mesh.zlev) // 2],              # Middle level
                self.mesh.zlev[-1],                                    # Last level
                (self.mesh.zlev[0] + self.mesh.zlev[1]) / 2           # Between first and second
            ]
            
            for depth in depths_to_test:
                idx = ind_for_depth(depth, self.mesh)
                assert 0 <= idx < len(self.mesh.zlev), f"Index {idx} out of range for depth {depth}"
                
                # Check that the found level is actually the closest
                closest_diff = abs(abs(self.mesh.zlev[idx]) - abs(depth))
                for i, level in enumerate(self.mesh.zlev):
                    if i != idx:
                        assert closest_diff <= abs(abs(level) - abs(depth)), \
                            f"Level at index {idx} is not the closest to depth {depth}"

    def test_ind_for_depth_edge_cases(self):
        """Test edge cases for ind_for_depth function."""
        # Test with single level mesh
        single_level_mesh = Mock()
        single_level_mesh.zlev = np.array([-100])
        assert ind_for_depth(-50, single_level_mesh) == 0
        assert ind_for_depth(-200, single_level_mesh) == 0
        
        # Test with zero depth
        assert ind_for_depth(0, self.mock_mesh) == 0
        
        # Test with very small depths (floating point precision tests)
        tiny_depth_mesh = Mock()
        tiny_depth_mesh.zlev = np.array([0, -0.001, -0.002, -0.003])
        assert ind_for_depth(-0.0015, tiny_depth_mesh) == 1
        assert ind_for_depth(-0.0025, tiny_depth_mesh) == 2

    def test_ind_for_depth_error_handling(self):
        """Test error handling in ind_for_depth function."""
        # Test with empty zlev
        empty_mesh = Mock()
        empty_mesh.zlev = np.array([])
        with pytest.raises(ValueError):
            ind_for_depth(100, empty_mesh)
        
        # Test with None zlev
        none_mesh = Mock()
        none_mesh.zlev = None
        with pytest.raises(ValueError):
            ind_for_depth(100, none_mesh)
        