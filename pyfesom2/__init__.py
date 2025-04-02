# -*- coding: utf-8 -*-
#
# This file is part of pyfesom2
# Original code by Dmitry Sidorenko, Qiang Wang, Sergey Danilov and Patrick Scholz
#

"""Top-level package for pyfesom2."""

__author__ = """FESOM team"""
__email__ = "koldunovn@gmail.com"
__version__ = "0.4.0"

from .accessor import FESOMDataArray as _FESOMDataArray
from .ascii_to_netcdf import read_fesom_ascii_grid, write_mesh_to_netcdf
from .climatology import *
from .datasets import open_dataset
from .diagnostics import *
from .fesom2GeoFormat import *
from .load_mesh_data import *
from .plotting import *
from .regridding import *
from .transect import *
from .transport import cross_section_transport
from .ut import *
