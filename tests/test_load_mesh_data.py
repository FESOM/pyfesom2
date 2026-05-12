#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for mesh class modernization in load_mesh_data module."""

import os
import warnings

import numpy as np
import pytest

from pyfesom2.load_mesh_data import (
    CYCLIC_ELEMENT_THRESHOLD,
    EARTH_RADIUS,
    LONGITUDE_PERIOD,
    LONGITUDE_WRAP_THRESHOLD,
    Mesh,
    fesom_mesh,
    ind_for_depth,
    load_mesh,
)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MESH_PATH = os.path.join(THIS_DIR, "data", "pi-grid")


@pytest.fixture
def mesh_instance():
    """Fixture providing a mesh instance for testing."""
    if not os.path.exists(MESH_PATH):
        pytest.skip(f"Test mesh data not found at {MESH_PATH}")
    return load_mesh(MESH_PATH, usepickle=False, usejoblib=False)


class TestConstants:
    """Test that module-level constants are defined correctly."""

    def test_earth_radius(self):
        """Test EARTH_RADIUS constant is defined."""
        assert EARTH_RADIUS == 6371000.0
        assert isinstance(EARTH_RADIUS, float)

    def test_longitude_wrap_threshold(self):
        """Test LONGITUDE_WRAP_THRESHOLD constant is defined."""
        assert LONGITUDE_WRAP_THRESHOLD == 355
        assert isinstance(LONGITUDE_WRAP_THRESHOLD, int)

    def test_longitude_period(self):
        """Test LONGITUDE_PERIOD constant is defined."""
        assert LONGITUDE_PERIOD == 360
        assert isinstance(LONGITUDE_PERIOD, int)

    def test_cyclic_element_threshold(self):
        """Test CYCLIC_ELEMENT_THRESHOLD constant is defined."""
        assert CYCLIC_ELEMENT_THRESHOLD == 100
        assert isinstance(CYCLIC_ELEMENT_THRESHOLD, int)


class TestMeshClass:
    """Test the new Mesh class."""

    def test_mesh_class_exists(self):
        """Test that Mesh class is available."""
        assert Mesh is not None

    def test_mesh_instantiation(self, mesh_instance):
        """Test that Mesh can be instantiated."""
        assert isinstance(mesh_instance, Mesh)

    def test_mesh_attributes(self, mesh_instance):
        """Test that Mesh has expected attributes."""
        assert hasattr(mesh_instance, "path")
        assert hasattr(mesh_instance, "x2")
        assert hasattr(mesh_instance, "y2")
        assert hasattr(mesh_instance, "n2d")
        assert hasattr(mesh_instance, "e2d")
        assert hasattr(mesh_instance, "elem")
        assert hasattr(mesh_instance, "nlev")
        assert hasattr(mesh_instance, "zlev")
        assert hasattr(mesh_instance, "alpha")
        assert hasattr(mesh_instance, "beta")
        assert hasattr(mesh_instance, "gamma")

    def test_mesh_n2d_value(self, mesh_instance):
        """Test that mesh has correct number of 2d nodes."""
        assert mesh_instance.n2d == 3140

    def test_mesh_e2d_value(self, mesh_instance):
        """Test that mesh has correct number of 2d elements."""
        assert mesh_instance.e2d == 5839

    def test_mesh_coordinates_shape(self, mesh_instance):
        """Test that coordinate arrays have correct shape."""
        assert len(mesh_instance.x2) == mesh_instance.n2d
        assert len(mesh_instance.y2) == mesh_instance.n2d

    def test_mesh_elements_shape(self, mesh_instance):
        """Test that element array has correct shape."""
        assert mesh_instance.elem.shape == (mesh_instance.e2d, 3)

    def test_mesh_default_abg(self):
        """Test that default alpha, beta, gamma are applied."""
        if not os.path.exists(MESH_PATH):
            pytest.skip(f"Test mesh data not found at {MESH_PATH}")
        mesh = Mesh(MESH_PATH)
        # Default from docstring is [50, 15, -90] but __init__ default is [50, 15, -90]
        assert mesh.alpha == 50
        assert mesh.beta == 15
        assert mesh.gamma == -90

    def test_mesh_custom_abg(self):
        """Test that custom alpha, beta, gamma can be set."""
        if not os.path.exists(MESH_PATH):
            pytest.skip(f"Test mesh data not found at {MESH_PATH}")
        mesh = Mesh(MESH_PATH, abg=[0, 0, 0])
        assert mesh.alpha == 0
        assert mesh.beta == 0
        assert mesh.gamma == 0

    def test_mesh_invalid_path(self):
        """Test that invalid path raises IOError."""
        with pytest.raises(IOError, match="does not exists"):
            Mesh("/nonexistent/path")

    def test_mesh_str_representation(self, mesh_instance):
        """Test that mesh has string representation."""
        mesh_str = str(mesh_instance)
        assert "FESOM mesh" in mesh_str
        assert "path" in mesh_str
        assert str(mesh_instance.n2d) in mesh_str
        assert str(mesh_instance.e2d) in mesh_str

    def test_mesh_repr_representation(self, mesh_instance):
        """Test that mesh has repr representation."""
        mesh_repr = repr(mesh_instance)
        assert "FESOM mesh" in mesh_repr


class TestDeprecatedFesomMesh:
    """Test the deprecated fesom_mesh class."""

    def test_fesom_mesh_exists(self):
        """Test that fesom_mesh class still exists."""
        assert fesom_mesh is not None

    def test_fesom_mesh_is_subclass(self):
        """Test that fesom_mesh inherits from Mesh."""
        assert issubclass(fesom_mesh, Mesh)

    def test_fesom_mesh_deprecation_warning(self):
        """Test that using fesom_mesh raises DeprecationWarning."""
        if not os.path.exists(MESH_PATH):
            pytest.skip(f"Test mesh data not found at {MESH_PATH}")

        with pytest.warns(DeprecationWarning, match="fesom_mesh is deprecated"):
            mesh = fesom_mesh(MESH_PATH, abg=[0, 0, 0])
            assert isinstance(mesh, Mesh)

    def test_fesom_mesh_functionality(self):
        """Test that fesom_mesh still works correctly."""
        if not os.path.exists(MESH_PATH):
            pytest.skip(f"Test mesh data not found at {MESH_PATH}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mesh = fesom_mesh(MESH_PATH, abg=[0, 0, 0])
            assert mesh.n2d == 3140
            assert mesh.e2d == 5839


class TestLoadMeshFunction:
    """Test the load_mesh function with new Mesh class."""

    def test_load_mesh_returns_mesh_instance(self):
        """Test that load_mesh returns a Mesh instance."""
        if not os.path.exists(MESH_PATH):
            pytest.skip(f"Test mesh data not found at {MESH_PATH}")

        mesh = load_mesh(MESH_PATH, usepickle=False, usejoblib=False)
        assert isinstance(mesh, Mesh)

    def test_load_mesh_no_pickle_no_joblib(self):
        """Test load_mesh without pickle or joblib."""
        if not os.path.exists(MESH_PATH):
            pytest.skip(f"Test mesh data not found at {MESH_PATH}")

        mesh = load_mesh(MESH_PATH, usepickle=False, usejoblib=False)
        assert mesh.n2d == 3140
        assert mesh.e2d == 5839

    def test_load_mesh_with_custom_abg(self):
        """Test load_mesh with custom abg parameters."""
        if not os.path.exists(MESH_PATH):
            pytest.skip(f"Test mesh data not found at {MESH_PATH}")

        mesh = load_mesh(MESH_PATH, abg=[0, 0, 0], usepickle=False, usejoblib=False)
        assert mesh.alpha == 0
        assert mesh.beta == 0
        assert mesh.gamma == 0


class TestIndForDepth:
    """Test the ind_for_depth helper function."""

    def test_ind_for_depth_basic(self, mesh_instance):
        """Test ind_for_depth finds closest depth index."""
        # Get a depth that exists in the mesh
        if len(mesh_instance.zlev) > 0:
            target_depth = abs(mesh_instance.zlev[0])
            index = ind_for_depth(target_depth, mesh_instance)
            assert isinstance(index, int)
            assert 0 <= index < mesh_instance.nlev

    def test_ind_for_depth_exact_match(self, mesh_instance):
        """Test ind_for_depth with exact depth match."""
        if len(mesh_instance.zlev) > 0:
            exact_depth = abs(mesh_instance.zlev[0])
            index = ind_for_depth(exact_depth, mesh_instance)
            assert abs(abs(mesh_instance.zlev[index]) - exact_depth) < 1e-6

    def test_ind_for_depth_intermediate(self, mesh_instance):
        """Test ind_for_depth finds nearest for intermediate depth."""
        if len(mesh_instance.zlev) > 1:
            # Create a depth between first and second level
            depth1 = abs(mesh_instance.zlev[0])
            depth2 = abs(mesh_instance.zlev[1])
            intermediate_depth = (depth1 + depth2) / 2
            index = ind_for_depth(intermediate_depth, mesh_instance)
            # Should return one of the two closest indices
            assert index in [0, 1]


class TestMeshComputations:
    """Test computed attributes of the mesh."""

    def test_voltri_computed(self, mesh_instance):
        """Test that triangle volumes are computed."""
        assert hasattr(mesh_instance, "voltri")
        assert len(mesh_instance.voltri) == mesh_instance.e2d
        # All volumes should be positive
        assert np.all(mesh_instance.voltri > 0)

    def test_lump2_computed(self, mesh_instance):
        """Test that 2D lump operator is computed."""
        assert hasattr(mesh_instance, "lump2")
        assert len(mesh_instance.lump2) == mesh_instance.n2d
        # All values should be positive
        assert np.all(mesh_instance.lump2 > 0)

    def test_no_cyclic_elem_computed(self, mesh_instance):
        """Test that non-cyclic elements are identified."""
        assert hasattr(mesh_instance, "no_cyclic_elem")
        # All indices should be valid element indices
        assert np.all(mesh_instance.no_cyclic_elem >= 0)
        assert np.all(mesh_instance.no_cyclic_elem < mesh_instance.e2d)

    def test_topology_loaded(self, mesh_instance):
        """Test that topology information is loaded."""
        assert hasattr(mesh_instance, "topo")
        assert len(mesh_instance.topo) == mesh_instance.n2d
