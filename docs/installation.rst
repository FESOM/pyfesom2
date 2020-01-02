.. highlight:: shell

============
Installation
============

Now we support only installation from source and recomend to use `conda <https://conda.io/docs/>`_  to install dependencies. The shortest way to succes consist of the following simple steps:

1. Go to `Miniconda <https://conda.io/miniconda.html>`_ website and download Miniconda installation script for your system. We recomend to use python 3.7 version.


2. Install `Miniconda`. Don't forget to add the path of `Miniconda` instllation to your `$PATH` and relaunch your terminal.


3. Add conda-forge channel::

    conda config --add channels conda-forge


4. Go to the folder where you want to have pyfesom and execute (you have to have git installed)::

    git clone https://github.com/FESOM/pyfesom2.git

5. Go to `pyfesom2` folder and create conda environment from file::

    conda env create -f ./ci/requirements-py37.yml


6. Activate `pyfesom2` conda environment::

    conda activate pyfesom2

7. While in pyfesom2 folder execute::

    pip install -e .

this will install pyfesom2 in to your conda pyfesom2 environment.

Now when you want to use pyfesom2 as a library or one of the pyfesom2 tools, you have to execute::

    conda activate pyfesom2


