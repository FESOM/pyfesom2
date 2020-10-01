.. highlight:: shell

============
Installation
============

The best way to install now is to use `conda`. The short guide how to install it can be found for `Linux/Mac <https://github.com/koldunovn/python_for_geosciences/blob/master/README.md#getting-started-for-linuxmac>`_  and `Windows <https://github.com/koldunovn/python_for_geosciences/blob/master/README.md#getting-started-for-windows>`_ . For now we are not testing for Windows, and there is no garantee that pyfesom2 will work on this OS.

The easiest way is to install latest stable version from `conda-forge`::

    conda config --add channels conda-forge
    conda install pyfesom2

************************
Development Installation
************************

If you want to use the latest version of the code, or just plan to change the code inside the package, you have to install it in "development" mode. After you install `conda` (python 3.7 environment is recomended), clone the source code::

    git clone https://github.com/FESOM/pyfesom2.git

Create pyfesom2 environment by::

    conda env create -f ./pyfesom2/ci/requirements-py37.yml
    
Activate the environment::

    conda activate pyfesom2
    
Install pyfesom2::

    cd pyfesom2
    pip install -e .



