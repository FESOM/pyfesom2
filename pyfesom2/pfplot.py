# -*- coding: utf-8 -*-

import argparse
import matplotlib.pylab as plt

def pfplot():
    parser = argparse.ArgumentParser(prog='pfplot', 
                                     description='Plot FESOM2 data on the map.')
    parser.add_argument('meshpath',      help='Path to the mesh folder')
    parser.add_argument('ifile',         help='Path to the netCDF FESOM2 file')
    parser.add_argument('variable',      default='temp', help='Name of the variable inside the file')
    parser.add_argument('--depth', '-d', default=0, type=float, help='Depth in meters.')
    parser.add_argument('--box', '-b',   nargs=4, type=float, default=[-180., 180., -80., 90.], help='Map boundaries in -180 180 -90 90 format that will be used for interpolation.', metavar=('LONMIN', 'LONMAX', 'LATMIN', 'LATMAX'))
    parser.add_argument('--res', '-r',  nargs=2, type=int, default=(360, 170), help='Number of points along each axis that will be used for interpolation (for lon and  lat).', metavar=('N_POINTS_LON', 'N_POINTS_LAT'))
    parser.add_argument('--influence','-i', default=80000, type=float, help='Radius of influence for interpolation, in meters.')
    parser.add_argument('--timestep', '-t', default=0, type=int, help='Index of the timstep from netCDF variable, strats with 0.')
    parser.add_argument('--levels', '-l', nargs=3, type=float, help='Levels for contour plot in format min max numberOfLevels.\
    If not provided min/max values from data will be used with 40 levels.', metavar=('START', 'STOP', 'NUMBER' ))
    parser.add_argument('--quiet', '-q', action='store_true',  help='If present additional information will not be printed.')
    parser.add_argument('--ofile', '-o', type=str, help='Path to the output figure. If present the image\
                        will be saved to the file instead of showing it. ')
    parser.add_argument('--mapproj','-m', choices=['merc', 'pc', 'np', 'sp', 'rob'], default='rob',
                        help = 'Map projection. Options are Mercator (merc), Plate Carree (pc), North Polar Stereo (np), South Polar Stereo (sp),  Robinson (rob)')
    parser.add_argument('--abg', nargs=3, type=float, default=(0., 0., 0.),
              help='Alpha, beta and gamma Euler angles. If you plots look rotated, you use wrong abg values. Usually nessesary only during the first use of the mesh.')
    parser.add_argument('--clim','-c', type=str,
              help='Path to the file with climatology. If option is set the model bias to climatology will be shown.')
    parser.add_argument('--cmap', help='Name of the colormap from cmocean package or from the standard matplotlib set. By default `Spectral_r` will be used for property plots and `balance` for bias plots.')
    parser.add_argument('--interp', choices=['nn', 'idist', 'linear', 'cubic'],
                        default='nn',
                        help = 'Interpolation method. Options are nn - nearest neighbor (KDTree implementation, fast), idist - inverse distance (KDTree implementation, decent speed), linear (scipy implementation, slow) and cubic (scipy implementation, slowest and give strange results on corarse meshes).')
    parser.add_argument('--ptype', choices=['cf', 'pcm'], default = 'cf',
                        help = 'Plot type. Options are contourf (\'cf\') and pcolormesh (\'pcm\')')
    parser.add_argument('-k', type=int, default = 1,
                        help ='k-th nearest neighbors to use. Only used when interpolation method (--interp) is idist')

    args = parser.parse_args()
    # args.func(args)
    if not args.quiet:
        print("Mesh path:                     {}".format(args.meshpath))
        print("Input file path:               {}".format(args.ifile))
        print("Name of the variable:          {}".format(args.variable))
        print("Depth:                         {}".format(args.depth))
        print("Bounding box:                  {}".format(args.box))
        print("Number of points along sides:  {}".format(args.res))
        print("Radius of influence (in m.):   {}".format(args.influence))
        print("Nearest neighbors to use:      {}".format(args.k))
        print("Timestep index:                {}".format(args.timestep))
        print("Contour plot levels:           {}".format(args.levels))
        print("Quiet?:                        {}".format(args.quiet))
        print("Output file for image:         {}".format(args.ofile))
        print("Map projection:                {}".format(args.mapproj))
        print("Euler angles of mesh rotation: {}".format(args.abg))
        print("File with climatology:         {}".format(args.clim))
        print("Name of the color map:         {}".format(args.cmap))
        print("Interpolation method:          {}".format(args.interp))
        print("Plot type:                     {}".format(args.ptype))

    if args.cmap:
        if args.cmap in cmo.cmapnames:
            colormap = cmo.cmap_d[cmap]
        elif args.cmap in plt.cm.datad:
            colormap = plt.get_cmap(args.cmap)
        else:
            raise ValueError('Get unrecognised name for the colormap `{}`. Colormaps should be from standard matplotlib set of from cmocean package.'.format(args.cmap))
    else:
        if args.clim:
            colormap = cmo.cmap_d['balance']
        else:
            colormap = plt.get_cmap('Spectral_r')
    
    



if __name__ == '__main__':
    pfplot()