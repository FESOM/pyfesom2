# -*- coding: utf-8 -*-

import argparse
from collections import OrderedDict
import numpy as np
from .load_mesh_data import get_data, load_mesh, ind_for_depth
from .regriding import fesom2regular
from .ut import mask_ne, set_standard_attrs
import xarray as xr

def pfinterp():
    parser = argparse.ArgumentParser(
        prog="pfinterp", description="Interpolates FESOM2 data to regular grid."
    )
    parser.add_argument("meshpath", help="Path to the mesh folder")
    parser.add_argument("result_path", help="Path to the results")
    parser.add_argument(
        "variable", default="temp", help="Name of the variable inside the file"
    )
    parser.add_argument(
        "--years",
        "-y",
        default="1948",
        type=str,
        help="Years as a string. Options are one year, coma separated years, range in a form of 1948:2000 or * for everything.",
    )
    parser.add_argument(
        "--depths", "-d", default="0", type=str, help="Depths in meters. \
            Closest values from model levels will be taken.\
            Several options available: number - e.g. '100',\
                                       coma separated list - e.g. '0,10,100,200',\
                                       -1 - all levels will be selected."
    )
    parser.add_argument(
        "--box",
        "-b",
        nargs=4,
        type=float,
        default=[-180.0, 180.0, -80.0, 90.0],
        help="Map boundaries in -180 180 -90 90 format that will be used for interpolation.",
        metavar=("LONMIN", "LONMAX", "LATMIN", "LATMAX"),
    )
    parser.add_argument(
        "--res",
        "-r",
        nargs=2,
        type=int,
        default=(360, 170),
        help="Number of points along each axis that will be used for interpolation (for lon and  lat).",
        metavar=("N_POINTS_LON", "N_POINTS_LAT"),
    )
    parser.add_argument(
        "--influence",
        "-i",
        default=80000,
        type=float,
        help="Radius of influence for interpolation, in meters.",
    )
    parser.add_argument(
        "--timesteps",
        "-t",
        default="-1",
        type=str,
        help="Explicitly define timesteps of the input fields. There are several oprions:\
            '-1' - all time steps, number - one time step (e.g. '5'), numbers - coma separated (e.g. '0, 3, 8, 10'), slice - e.g. '5:10',\
            slice with steps - e.g. '8:-1:12'.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="If present additional information will not be printed.",
    )
    parser.add_argument(
        "--ofile",
        "-o",
        default="out.nc",
        type=str,
        help="Path to the output file. Default is ./out.nc.",
    )
    parser.add_argument(
        "--abg",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        help="Alpha, beta and gamma Euler angles. If you plots look rotated, you use wrong abg values. Usually nessesary only during the first use of the mesh.",
    )
    parser.add_argument(
        "--interp",
        choices=["nn", "idist", "linear", "cubic"],
        default="nn",
        help="Interpolation method. Options are nn - nearest neighbor (KDTree implementation, fast), idist - inverse distance (KDTree implementation, decent speed), linear (scipy implementation, slow) and cubic (scipy implementation, slowest and give strange results on corarse meshes).",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=1,
        help="k-th nearest neighbors to use. Only used when interpolation method (--interp) is idist",
    )

    args = parser.parse_args()
    # args.func(args)
    if not args.quiet:
        print("Mesh path:                     {}".format(args.meshpath))
        print("Input file path:               {}".format(args.result_path))
        print("Name of the variable:          {}".format(args.variable))
        print("Years:                         {}".format(args.years))
        print("Depths:                         {}".format(args.depths))
        print("Bounding box:                  {}".format(args.box))
        print("Number of points along sides:  {}".format(args.res))
        print("Radius of influence (in m.):   {}".format(args.influence))
        print("Nearest neighbors to use:      {}".format(args.k))
        print("Timesteps index:               {}".format(args.timesteps))
        print("Quiet?:                        {}".format(args.quiet))
        print("Output file:                   {}".format(args.ofile))
        print("Euler angles of mesh rotation: {}".format(args.abg))
        print("Interpolation method:          {}".format(args.interp))


    years = args.years
    if len(years.split(":")) == 2:
        y = range(int(years.split(":")[0]), int(years.split(":")[1]))
    elif len(years.split(",")) > 1:
        y = list(map(int, years.split(",")))
    else:
        y = [int(years)]
    years = y
    # args.timesteps = [0,1]

    timesteps = args.timesteps
    if len(timesteps.split(":")) == 2:
        y = slice(int(timesteps.split(":")[0]), int(timesteps.split(":")[1]))
    if len(timesteps.split(":")) == 3:
        y = slice(int(timesteps.split(":")[0]),
                  int(timesteps.split(":")[1]),
                  int(timesteps.split(":")[2]))
    elif len(timesteps.split(",")) > 1:
        y = list(map(int, timesteps.split(",")))
    elif int(timesteps) == -1:
        y = -1
    else:
        y = [int(timesteps)]
    timesteps = y
    print("timesteps {}".format(timesteps))

    mesh = load_mesh(args.meshpath, abg=args.abg, usepickle=True, usejoblib=False)

    depths = args.depths

    if len(depths.split(",")) > 1:
        depths = list(map(int, depths.split(",")))
    elif int(depths) == -1:
        depths = [-1]
    else:
        depths = [int(depths)]
    print(depths)

    if depths[0] == -1:
        dind = range(mesh.zlev.shape[0])
        realdepth = mesh.zlev
    else:
        dind = []
        realdepth = []
        for depth in depths:
            ddepth = ind_for_depth(depth, mesh)
            dind.append(ddepth)
            realdepth.append(mesh.zlev[ddepth])
    print(dind)
    print(realdepth)

    data = get_data(
        result_path=args.result_path,
        variable=args.variable,
        years=years,
        mesh=mesh,
        runid="fesom",
        records=timesteps,
        depth=None,
        how=None,
        ncfile=None,
        compute=False,
    )
    if len(dind) <= data.shape[2]:
        data = data.isel(nz1=dind)
    elif len(dind) > data.shape[2]:
        dind = dind[:-1]
        realdepth = realdepth[:-1]
        data = data.isel(nz1=dind)

    left, right, down, up = args.box
    lonNumber, latNumber = args.res

    lonreg = np.linspace(left, right, lonNumber)
    latreg = np.linspace(down, up, latNumber)
    lonreg2, latreg2 = np.meshgrid(lonreg, latreg)

    dshape = data.shape
    empty_data = np.empty((dshape[0], dshape[2], latNumber, lonNumber ))

    da = xr.DataArray(empty_data, dims=['time', 'depth_coord', 'lat', 'lon'],
                          coords={'time':data.time,
                                  'depth_coord':realdepth,
                                  'lat':latreg2[:,0].flatten(),
                                  'lon':lonreg2[0,:].flatten()},
                                  name=args.variable,
                                  attrs=data.attrs)
    da = set_standard_attrs(da)
    m2 = mask_ne(lonreg2, latreg2)

    for timestep in range(da.time.shape[0]):
        for depth_ind in range(da.depth_coord.shape[0]):
            interp_data = fesom2regular(
                                        data[timestep,:,depth_ind].values,
                                        mesh,
                                        lonreg2,
                                        latreg2,
                                        distances_path=None,
                                        inds_path=None,
                                        qhull_path=None,
                                        how=args.interp,
                                        k=args.k,
                                        radius_of_influence=args.influence,
                                        n_jobs=2,
                                        dumpfile=True,
                                        basepath=None,
                                    )
            interp_data = np.ma.masked_where(m2, interp_data)
            interp_data = np.ma.masked_equal(interp_data, 0)
            da[timestep, depth_ind,:,:] = interp_data[:]

    da.to_netcdf(args.ofile)


# parser.set_defaults(func=pfinterp)


if __name__ == "__main__":
    # args = parser.parse_args()
    # args.func(args)
    pfinterp()
