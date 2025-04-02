# -*- coding: utf-8 -*-
#
# This file is part of pyfesom2
# Original code by Dmitry Sidorenko, Qiang Wang, Sergey Danilov and Patrick Scholz
#

"""Top-level package for pyfesom2."""

__author__ = """FESOM team"""
__email__ = "koldunovn@gmail.com"
__version__ = "0.4.1"

from .load_mesh_data import *
from .plotting import *
from .climatology import *
from .regridding import *
from .transect import *
from .diagnostics import *
from .ut import *
from .fesom2GeoFormat import *
from .datasets import open_dataset
from .accessor import FESOMDataArray as _FESOMDataArray
from .transport import cross_section_transport
from .ascii_to_netcdf import read_fesom_ascii_grid, write_mesh_to_netcdf
