#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pyfesom2.ascii_to_netcdf`."""

import os
from collections import Counter

import numpy as np
import pytest
from netCDF4 import Dataset

from pyfesom2 import read_fesom_ascii_grid, write_mesh_to_netcdf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GRIDDIR = os.path.join(THIS_DIR, "data", "pi-grid")


@pytest.fixture(scope="module")
def grid():
    return read_fesom_ascii_grid(griddir=GRIDDIR, verbose=False)


def test_coast_matches_boundary_edges(grid):
    # a node is coastal iff it lies on an edge that belongs to one element only
    elem = grid["elem"]
    edges = Counter()
    for e in elem:
        for a, b in ((e[0], e[1]), (e[1], e[2]), (e[2], e[0])):
            edges[(min(a, b), max(a, b))] += 1
    boundary = np.zeros(grid["N"], dtype=bool)
    for (a, b), n in edges.items():
        if n == 1:
            boundary[a - 1] = boundary[b - 1] = True
    assert boundary.any()
    np.testing.assert_array_equal(np.asarray(grid["coast"], dtype=bool), boundary)


def test_neighbours_are_ordered_around_node(grid):
    # consecutive neighbours j, j+1 of node i share the element listed between them
    elem = grid["elem"]
    neighnodes, neighelems = grid["neighnodes"], grid["neighelems"]
    Nneighs = np.sum(~np.isnan(neighnodes), axis=1)
    for i in range(grid["N"]):
        for j in range(Nneighs[i] - 1):
            tri = set(elem[int(neighelems[i, j])])
            assert tri == {i + 1, int(neighnodes[i, j]), int(neighnodes[i, j + 1])}


def test_vertical_levels_match_mesh_diag(grid, tmp_path):
    ofile = str(tmp_path / "mesh.nc")
    write_mesh_to_netcdf(grid, ofile=ofile, overwrite=True, verbose=False)
    with Dataset(ofile) as m, Dataset(os.path.join(GRIDDIR, "fesom.mesh.diag.nc")) as d:
        assert len(m.dimensions["nlev"]) == len(d.dimensions["nz1"])
        np.testing.assert_allclose(m["depth_bnds"][:], np.abs(d["nz"][:]))
        np.testing.assert_allclose(m["depth"][:], np.abs(d["nz1"][:]))
        # number of layers = 1-based index of the bottom interface - 1
        np.testing.assert_array_equal(m["depth_lev"][:], d["nlevels_nod2D"][:] - 1)
