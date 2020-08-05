#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "cartopy",
    "cmocean",
    "dask",
    "ipython",
    "joblib",
    "jupyter",
    "matplotlib",
    "netCDF4",
    "numba",
    "numpy",
    "pandas",
    "pyresample",
    "pytest",
    "scipy",
    "seawater",
    "shapely",
    "xarray",
]

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest"]

setup(
    author="FESOM team",
    author_email="koldunovn@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    entry_points={
        "console_scripts": [
            "pfplot=pyfesom2.pfplot:pfplot",  # command=package.module:function
            "pfinterp=pyfesom2.pfinterp:pfinterp",
        ]
    },
    description="FESOM2 tools",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="pyfesom2",
    name="pyfesom2",
    packages=find_packages(include=["pyfesom2"]),
    package_dir={"pyfesom2": "pyfesom2"},
    package_data={"": ["*.geojson"]},
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/koldunovn/pyfesom2",
    download_url="https://github.com/FESOM/pyfesom2/archive/0.2.0.tar.gz",
    version="0.2.0",
    zip_safe=False,
)
