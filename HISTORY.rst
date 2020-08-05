=======
History
=======

0.2.0 (2020-08-05)
------------------

Breaking changes
----------------

* depricate ``plot_transect`` and ``hofm_plot`` functions. Replaced with more general ``plot_xyz`` function.
* replace ``transect_uv`` with ``get_transect_uv``.
* remove Basemap dependency and all code related to Basemap.

New Features
------------

* ``plot_vector`` - plotting vector on a map.
* ``plot_xyz`` - ploting, for example, transect or hovmoeller diagrams.
* ``get_transect`` - returns the data for transect, that then should be used with ``plot_xyz`` to plot.
* ``tplot`` - plotting on original FESOM mesh, without interpolation.
* ``compute_face_coords`` - compute coordinates (centers) of elements (triangles).
* ``cut_region`` - cut region from the mesh.
* ``xmoc_data`` - compute moc for selected region
* ``get_mask`` - create mask of the region (e.g. Atlantic or Pacific Ocean).

Bug fixes
---------

* fix ``get_cmap`` to cpmply with new matplotlib versions.
* fix RTD builds on master (by `Suvarchal Kumar Cheedela <https://github.com/suvarchal>`_).
* fix issue with time module change after python 3.7 (by `Paul Gierz <https://github.com/pgierz>`_)

Documentation
-------------

* Notebook example for vector plotting.
* Notebook example for plotting on original mesh with``tplot``.
* Notebook example for very fast plotting with `geoviews <https://geoviews.org/>`_ .
* Notebook example for xMOC plotting.

Internal Changes
----------------

* general cleaning up of the code
* split ``plot`` function to make it more readable.
* refactor ``tonodes`` function, allow xarray in ``tonodes3d``
* add tests for transects.

0.1.1 (2019-02-12)
------------------

* Fixes for PyPI

0.1.0 (2019-02-12)
------------------

* First release on PyPI.
