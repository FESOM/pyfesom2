pyfesom2
========
 workflow test

[![Build Status](https://travis-ci.org/koldunovn/pyfesom2.svg?branch=master)](https://travis-ci.org/koldunovn/pyfesom2)
[![Documentation Status](https://readthedocs.org/projects/pyfesom2/badge/?version=latest)](https://pyfesom2.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/FESOM/pyfesom2/HEAD)

FESOM2 tools


* Free software: MIT license
* Documentation: https://pyfesom2.readthedocs.io.


Installation
------------
Currently the easiest way is to use `conda`. The short guide how to install it can be found for [Linux/Mac](https://github.com/koldunovn/python_for_geosciences/blob/master/README.md#getting-started-for-linuxmac) and [Windows](https://github.com/koldunovn/python_for_geosciences/blob/master/README.md#getting-started-for-windows). For now we are not testing for Windows, and there is no garantee that pyfesom2 will work on this OS. 

The easiest way is to install latest stable version from `conda-forge`:

    conda config --add channels conda-forge
    conda install pyfesom2

Development Installation
------------------------
If you plan to change the code inside the package, you have to install it in "development" mode. For this you would also need a working `conda`. The short guide how to install it can be found for [Linux/Mac](https://github.com/koldunovn/python_for_geosciences/blob/master/README.md#getting-started-for-linuxmac) and [Windows](https://github.com/koldunovn/python_for_geosciences/blob/master/README.md#getting-started-for-windows).
After you install `conda` (python 3.7 environment is recomendes), clone the source code:


    git clone https://github.com/FESOM/pyfesom2.git


Create pyfesom2 environment by:

    conda env create -f ./pyfesom2/ci/requirements-py37.yml


Activate the environment

    conda activate pyfesom2


Install pyfesom2

    cd pyfesom2
    pip install -e .

Troubleshooting
---------------
There is a bug in cartopy, that lead to strange looking figures - giving you instead of land fill a river network:
![image](https://github.com/FESOM/pyfesom2/assets/3407313/d159c3bb-2ec3-4369-8eb6-ba9c905591cc)

To fix it one should download GSHHS files by hand, from [here](https://swift.dkrz.de/v1/dkrz_c719fbc3-98ea-446c-8e01-356dac22ed90/cartopy/shapefiles.tar) and untar them into `~.local/share/cartopy/shapefiles/` directory. As a result your plotting scripts should produce something like this:

![image](https://github.com/FESOM/pyfesom2/assets/3407313/4565c334-2d63-4560-b357-dd7af4d34a64)


Usage of tools
--------------
Below are couple of examples of CLI tools usage. 
For now two tools are available: `pfplot` (plot variable on the map) and `pfinterp` (interpolate scalar values to regular grid). You can get complete list of options by executing:

    pfplot -h


To plot temperature field one can do:

    pfplot ./CORE_MESH/ ./CORE_out/ temp


where `./CORE_MESH` is path to the mesh, `./CORE_out/` is path to the folder with results and `temp` is the name of the variable. By defauld `pfplot` will try to plot the mean values from the year 1948 at the surface (0 depth) on a global 360/180 map. Make sure you have write permissions to the folder with the mesh, since pfplot will save interpolation information, so next time it is not calculated but just loaded from the file.

To plot the different year:

    pfplot ./CORE_MESH/ ./CORE_out/ temp -y 2000


Plot different depth:

    pfplot ./CORE_MESH/ ./CORE_out/ temp -y 2000 -d 100


Plot mean over several years:

    pfplot ./CORE_MESH/ ./CORE_out/ temp -y 1948,1949,1950


Plot mean over several years by specifying the range:

    pfplot ./CORE_MESH/ ./CORE_out/ temp -y 1948:1955


Plot with North Polar Stereo projection (you have to specify different bounding box with `-b`)

    pfplot ./CORE_MESH/ ./CORE_out/ temp -y 2000 -b -180 180 60 90 -m np


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
