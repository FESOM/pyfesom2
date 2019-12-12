.. _pfinterp:

pfinterp
========

Interpolates scalar and vector data from FESOM mesh to regular lon/lat grid.

Basic usage
-----------

As a minimum you should provide path to the mesh, path to the file, path were the ouptut will be stored and variable name::

    pfinterp  /path/to/mesh/ /path/to/datafolder/ temp


by default `pfinterp` will search for the year 1948 and interpolate first time step from the depth 0 to 1 degree lon/lat regular grid.

In the following we just going to replace paths by shell variables::

    MESH=/path/to/mesh/
    DATA=/path/to/datafolder/

to make examples more consise. It is also a good practice to setup such variables for yourself, so the commands are shorter.

The resolution of the target grid is controlled by `-r` option, that accepts 2 arguments - number of longitudes and number of latitudes. For example to interpolate to the 1/4 degree grid for the box in the North Atlantic (defined by `-b` option), you should do the following::

    pfinterp $MESH $DATA temp -b -90 0 20 60 -r 360 160


If you do such interpolation for FESOM results on the COREII mesh and open the resulting file in ncview, it will look like this:

.. image:: img/pfinterp1.png

Example of a query with most common options::

    pfinterp $MESH $DATA temp -y 1950:1955 -d 0,100,500 -r 1440 720 -i 40000 --interp idist -k 5 -o /path/to/output.nc


This will result in interpolation of data from 6 years, for depths 0, 100 and 500 meters, with 1/4 degree resolution (1440x720 points), with radius of influence (characterisitc needed for nearest neighbor and inverse distance interpolations) of 40000 meters (40 km). The `--interp` sets `idist` (inverse distance) interpolation and with `-k` option you provide the number of neighboring points that will be used for interpolation. The `-o` sets the path to output `netCDF` file.

Below we look at some of the options in more detail.

Select time
-----------

You probably not always want to look at the default first timestep of the year 1948. There are several ways to select time intervals you would like to interpolate. First we can select different year (`-y` option)::

    pfinterp $MESH $DATA temp -y 1950

We also can select range of years::

    pfinterp $MESH $DATA temp -y 1950:1955

You will get one file with 6 fields (from 1950 to 1955 included). You can select specific years::

    pfinterp $MESH $DATA temp -y 1950,1955,1959

OK selecting the complete years is fine, but what if we need only some specific timesteps from the years we are selecting? Unfortunatelly for now you have to know how many timesteps you have selected with your year selection. To make it simplier here are couple of examples of year selection, time frequency of the data stored in each file and resulting number of timesteps:

==============     =========  =========
years              frequency  timesteps
--------------     ---------  ---------
1950                 yearly     1
1950:1959            yearly    10
1950:1959            monthly   120
1950,1952,1959       monthly   36
==============     =========  =========

To select one time step, you can just do::

    pfinterp $MESH $DATA temp -y 1950:1955 -t 5

this will select 5th timestep, but **THE COUNTING STARTS AT 0**, so for you files have data with monthly frequency, you will get interpolated field for **JUNE**.

You can select several timesteps (**no spaces between values!**)::

    pfinterp $MESH $DATA temp -y 1950:1955 -t 5,7,10

This particular case will return June, August ans November in case of monthly data.

You can select a slice of timesteps::

    pfinterp $MESH $DATA temp -y 1950:1955 -t 11:14

This, in case of monthly data, will select Dacember from the year 1950, and January, February from the year 1951. In contrast to years, **the last number** (in this case 14) **is not included**, similar to python slices.

Probably the most useful use of timestep selection, is when you have to exctract some month from the long line of monthly data. You can do it by providing slices with steps, like this::

    pfinterp $MESH $DATA temp -y 1950:1955 -t 8:72:12

This will return all timesteps starting from number 8 (September, remember we start to count from 0) untill the timestep 72 with step 12. So you will get all Septembers from years 1950 to 1955. But you don't have to always remember how many timesteps you have, just put the `end` instead of your last index::

    pfinterp $MESH $DATA temp -y 1950:1955 -t 8:end:12



You can clearly see imprint of the original mesh on the interpolated result. This is due to the nearest neighbor interpolation used by default. The advantage of this method is that it is very fast, but for some combunations of original and target grids can produce quite ugly results. There are several other interpolation methods, namelly `idist` (inverse distance, decent speed, `linear` (scipy implementation, slow), and `cubic` (scipy implementation, slowest and give strange results on corarse meshes). The default results for `idist` method will look like this:

