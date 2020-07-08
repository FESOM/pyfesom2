# -*- coding: utf-8 -*-
#
# This file is part of pyfesom2
# Original code by Dmitry Sidorenko, Nikolay Koldunov, 
# Qiang Wang, Sergey Danilov and Patrick Scholz
#

import argparse
from collections import OrderedDict

import numpy as np
import xarray as xr

from .load_mesh_data import get_data, ind_for_depth, load_mesh
from .regridding import fesom2regular, tonodes
from .ut import mask_ne, set_standard_attrs, vec_rotate_r2g


def parse_years(years):

    if len(years.split(":")) == 2:
        y = range(int(years.split(":")[0]), int(years.split(":")[1]) + 1)
    elif len(years.split(",")) > 1:
        y = list(map(int, years.split(",")))
    else:
        y = [int(years)]
    years = y
    return years


def parse_timesteps(timesteps, time_shape):

    if len(timesteps.split(":")) == 2:
        y = range(int(timesteps.split(":")[0]), int(timesteps.split(":")[1]))
        # y = slice(int(timesteps.split(":")[0]), int(timesteps.split(":")[1]))
    elif len(timesteps.split(":")) == 3:
        if timesteps.split(":")[1] == "end":
            stop = time_shape
        else:
            stop = int(timesteps.split(":")[1])
        y = range(int(timesteps.split(":")[0]), stop, int(timesteps.split(":")[2]))
        # y = slice(int(timesteps.split(":")[0]),
        #           int(timesteps.split(":")[1]),
        #           int(timesteps.split(":")[2]))
    elif len(timesteps.split(",")) > 1:
        y = list(map(int, timesteps.split(",")))
    elif int(timesteps) == -1:
        y = -1
    else:
        y = [int(timesteps)]
    timesteps = y
    print("timesteps {}".format(timesteps))
    return timesteps


def parse_depths(depths, mesh, vertical_type="nz1"):

    if len(depths.split(",")) > 1:
        depths = list(map(int, depths.split(",")))
    elif int(depths) == -1:
        depths = [-1]
    else:
        depths = [int(depths)]
    print(depths)

    if depths[0] == -1:
        if vertical_type == "nz":
            dind = range(mesh.zlev.shape[0])
            realdepth = mesh.zlev
        elif vertical_type == "nz1":
            dind = range(mesh.zlev.shape[0] - 1)
            realdepth = mesh.zlev[:-1]
        # 2d data, ignoring -1 option for depth
        else:
            dind = [0]
            realdepth = [0]
    else:
        dind = []
        realdepth = []
        for depth in depths:
            ddepth = ind_for_depth(depth, mesh)
            dind.append(ddepth)
            realdepth.append(mesh.zlev[ddepth])
    print(dind)
    print(realdepth)
    return dind, realdepth


def get_data_forint(result_path, variable, years, mesh, depth, timestep):
    vector_vars = {}
    vector_vars["u"] = ["u", "v"]
    vector_vars["v"] = ["u", "v"]
    vector_vars["uice"] = ["uice", "vice"]
    vector_vars["vice"] = ["uice", "vice"]
    vector_vars["u100"] = ["u100", "v100"]
    vector_vars["v100"] = ["u100", "v100"]
    vector_vars["u30"] = ["u30", "v30"]
    vector_vars["v30"] = ["u30", "v30"]

    if variable not in vector_vars:
        # usuall scalar variable, things as usual
        data = get_data(
            result_path=result_path,
            variable=variable,
            years=years,
            mesh=mesh,
            runid="fesom",
            records=-1,
            depth=depth,
            how=None,
            ncfile=None,
            compute=False,
        )
        data_forint = data[timestep, :].values
        if data_forint.shape[0] == mesh.e2d:
            data_forint = tonodes(
                data_forint.astype("float32"),
                mesh.n2d,
                mesh.voltri,
                mesh.elem,
                mesh.e2d,
                mesh.lump2,
            )

    else:
        # vector variabel, should be rotated.
        data_u = get_data(
            result_path=result_path,
            variable=vector_vars[variable][0],
            years=years,
            mesh=mesh,
            runid="fesom",
            records=-1,
            depth=depth,
            how=None,
            ncfile=None,
            compute=False,
        )
        data_v = get_data(
            result_path=result_path,
            variable=vector_vars[variable][1],
            years=years,
            mesh=mesh,
            runid="fesom",
            records=-1,
            depth=depth,
            how=None,
            ncfile=None,
            compute=False,
        )
        data_u_int = data_u[timestep, :].values
        data_v_int = data_v[timestep, :].values

        if variable in ["u", "v", "u100", "v100", "u30", "v30"]:

            u_nodes = tonodes(
                data_u_int.astype("float32"),
                mesh.n2d,
                mesh.voltri,
                mesh.elem,
                mesh.e2d,
                mesh.lump2,
            )
            v_nodes = tonodes(
                data_v_int.astype("float32"),
                mesh.n2d,
                mesh.voltri,
                mesh.elem,
                mesh.e2d,
                mesh.lump2,
            )
        else:
            u_nodes = data_u_int.astype("float32")
            v_nodes = data_v_int.astype("float32")

        uu, vv = vec_rotate_r2g(50, 15, -90, mesh.x2, mesh.y2, u_nodes, v_nodes, flag=1)
        if variable in ["u", "uice"]:
            data_forint = uu
        else:
            data_forint = vv
    return data_forint


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
        help="Years as a string. Options are one year, coma separated years, or range in a form of 1948:2000.",
    )
    parser.add_argument(
        "--depths",
        "-d",
        default="0",
        type=str,
        help="Depths in meters. \
            Closest values from model levels will be taken.\
            Several options available: number - e.g. '100',\
                                       coma separated list - e.g. '0,10,100,200',\
                                       -1 - all levels will be selected.",
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
            slice with steps - e.g. '8:120:12'.\
            slice untill the end of time series - e.g. '8:end:12'.",
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
        default=5,
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

    mesh = load_mesh(args.meshpath, abg=args.abg, usepickle=True, usejoblib=False)

    years = parse_years(args.years)

    # prepear mesh for interpolation
    left, right, down, up = args.box
    lonNumber, latNumber = args.res

    lonreg = np.linspace(left, right, lonNumber)
    latreg = np.linspace(down, up, latNumber)
    lonreg2, latreg2 = np.meshgrid(lonreg, latreg)

    # first load the metadata to get more information
    data = get_data(
        result_path=args.result_path,
        variable=args.variable,
        years=years,
        mesh=mesh,
        runid="fesom",
        records=-1,
        depth=None,
        how=None,
        ncfile=None,
        compute=False,
    )

    time_shape = data.time.shape[0]
    timesteps = parse_timesteps(args.timesteps, time_shape)

    # select all timesteps
    if timesteps == -1:
        timesteps = range(time_shape)
    # set timestep to 0 if data have only one time step
    if time_shape == 1:
        timesteps = [0]

    if "nz" in data.dims:
        dind, realdepth = parse_depths(args.depths, mesh, "nz")
    elif "nz1" in data.dims:
        dind, realdepth = parse_depths(args.depths, mesh, "nz1")
    else:
        dind, realdepth = parse_depths(args.depths, mesh, "2d")

    empty_data = np.empty((len(timesteps), len(dind), latNumber, lonNumber))

    da = xr.DataArray(
        empty_data,
        dims=["time", "depth_coord", "lat", "lon"],
        coords={
            "time": data.time[timesteps],
            "depth_coord": realdepth,
            "lat": latreg2[:, 0].flatten(),
            "lon": lonreg2[0, :].flatten(),
        },
        name=args.variable,
        attrs=data.attrs,
    )
    da = set_standard_attrs(da)
    m2 = mask_ne(lonreg2, latreg2)

    for timestep_index, timestep in enumerate(timesteps):
        for depth_index, depth_model in enumerate(realdepth):
            data = get_data_forint(
                result_path=args.result_path,
                variable=args.variable,
                years=years,
                mesh=mesh,
                depth=depth_model,
                timestep=timestep,
            )

            interp_data = fesom2regular(
                data,
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
            da[timestep_index, depth_index, :, :] = interp_data[:]

    da.to_netcdf(args.ofile)


if __name__ == "__main__":
    # args = parser.parse_args()
    # args.func(args)
    pfinterp()
