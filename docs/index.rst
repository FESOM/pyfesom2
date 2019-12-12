pyfesom2
========

Python library and collection of tools for basic handling of `FESOM2 <http://www.fesom.de/>`_  ocean model output.

:ref:`tools` are python scripts with command line interfaces that are used for quick actions with FESOM2 model output. For example::

    pfplot /path/to/mesh/ /path/to/output/ salt

will produce a map with global spatial distribution of salinity on the surface at the first time step.

Library is a python library that contains functions for working with FESOM2 mesh and data. For example loading FESOM mesh can be done as simple as::

    import pyfesom2 as pf
    meshpath  ='/path/to/mesh/'
    mesh = pf.load_mesh(meshpath)

Examples of tools are :ref:`pfplot` for quick visualization of FESOM data and :ref:`pfinterp` for interpolation to regular lon/lat grid.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tools
   library
   api
   contributing
   history

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
