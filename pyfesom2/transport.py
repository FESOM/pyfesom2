"""
Module for computing transports across sections from fesom2 output
Author: Finn Heukamp (finn.heukamp@awi.de)
Initial version: 23.12.2021
"""

import warnings
from os.path import isfile, isdir
import xarray as xr
import numpy as np
import shapely.geometry as sg
import pyproj
from dask.diagnostics import ProgressBar
ProgressBar().register()

from .load_mesh_data import load_mesh
from .ut import vec_rotate_r2g, get_no_cyclic, cut_region



def _ProcessInputs(section, mesh_path, data_path, mesh_diag_path, years, how, use_great_circle):
    '''
    process_inputs.py

    Processes inputs and aborts if inputs are wrong

    Inputs
    ------
    section (list, str)
        either a list of the form [lon_start, lon_end, lat_start, lat_end] or a string for a preset section
    mesh_path (str)
        directory where the mesh files are stored
    data_path (str)
        directory where the data is stored
    mesh_diag_path (str: optional, default=None)
        directory where the mesh_diag file is stored, if None it is assumed to be located in data_path

    Returns
    -------

        '''

    # Check the input data types
    if not isinstance(section, list) | isinstance(section, str):
        raise ValueError(
            'The section must be a list of form [lon_start, lon_end, lat_start, lat_end] or a string for a preset section ("FS", "BSO", "BSX", ...)')

    if isinstance(section, list) & (len(section) != 4):
        raise ValueError(
            'The section must be a list of form [lon_start, lon_end, lat_start, lat_end]')

    if not isinstance(mesh_path, str):
        raise ValueError('mesh path must be a string')

    if not isinstance(data_path, str):
        raise ValueError('data path must be a string')

    if (mesh_diag_path != None) & (not isinstance(mesh_diag_path, str)):
        raise ValueError('mesh diag path must be a string')

    if (how != 'ori') & (how != 'mean'):
        raise ValueError(
            'how must be either ori for all timesteps or mean for the time mwan velocity')

    # Check for existance of the files
    files_u = [data_path + 'u.fesom.' + str(year) + '.nc' for year in years]
    files_v = [data_path + 'v.fesom.' + str(year) + '.nc' for year in years]
    files = files_u + files_v

    file_check = []
    for file in files:
        file_check.append(isfile(file))

    if not all(file_check):
        raise FileExistsError('One or more of the velocity files do not exist!')

    if not isdir(mesh_path):
        raise FileExistsError('The mesh folder does not exist!')

    if mesh_diag_path == None:
        mesh_diag_path = data_path + 'fesom.mesh.diag.nc'
        if not isfile(mesh_diag_path):
            raise FileExistsError(
                'The fesom.mesh.diag.nc file is not located in data_path! Please specify the absolute path!')
    elif isinstance(mesh_diag_path, str):
        if not isfile(mesh_diag_path):
            raise FileExistsError('The mesh diag file does not exist!')

    # Load the mesh and the mesh_diag files
    mesh = load_mesh(mesh_path)
    mesh_diag = xr.open_dataset(mesh_diag_path)

    # Create the section dictionary
    if isinstance(section, list):
        section = {'lon_start': section[0],
                   'lon_end': section[1],
                   'lat_start': section[2],
                   'lat_end': section[3],
                   }
        section['name'] = 'not specified'

    elif isinstance(section, str):
        section_name = section

        presets = ["BSO", "BSX", "BEAR_SVAL", "SVAL_KVITOYA", "KVITOYA_FJL",
                   "ST_ANNA_THROUGH", "SVINOY", "GIMSOY", "FRAMSTRAIT"]
        if not section in presets:
            raise ValueError('The chosen preset section does not exist!')
        else:
            if section_name == 'BSO':
                section = {'lon_start': 19.999,
                           'lon_end': 19.999,
                           'lat_start': 70,
                           'lat_end': 74.5,
                           }

            elif section_name == 'BSX':
                section = {'lon_start': 64,
                           'lon_end': 64,
                           'lat_start': 76,
                           'lat_end': 81,
                           }

            elif section_name == 'FRAMSTRAIT':
                section = {'lon_start': -6,
                           'lon_end': 10,
                           'lat_start': 78.8,
                           'lat_end': 78.8,
                           }

            elif section_name == 'BEAR_SVAL':
                section = {'lon_start': 19.999,
                           'lon_end': 19.999,
                           'lat_start': 74.5,
                           'lat_end': 78,
                           }

            # add more presets here

        section['name'] = section_name

    # Find the orientation of the section
    if section['lon_start'] == section['lon_end']:
        section['orientation'] = 'meridional'
    elif (section['lat_start'] == section['lat_end']) & (use_great_circle == False):
        section['orientation'] = 'zonal'
    else:
        section['orientation'] = 'other'
        warnings.warn('The transport computation for non zonal or non meridional sections is experimental and \
                       no warranty for its correctness is given!')

    # Add great circle information
    if use_great_circle:
        section['great_circle'] = True
    else:
        section['great_circle'] = False
        warnings.warn('For zonal sections the length of the non-great-circle section is computed with a \
                       reference length of 111.568 km/°E * cos(lat)')


    # add year information
    section['years'] = years

    return mesh, mesh_diag, files, section


def _ComputeWaypoints(section, mesh, use_great_circle):
    '''
    compute_waypoints.py

    Computes the waypoints between the section start and end either along a great circle or linear

    Inputs
    ------
    section (dict)
        section dictionary containing the section start, end and orientation data
    mesh (fesom.mesh object)
        fesom.mesh
    use_great_circle (bool)
        True or False
    '''
    if use_great_circle:
        # Compute the great circle coordinates along the section
        g = pyproj.Geod(ellps='WGS84')

        section_waypoints = g.npts(section['lon_start'],
                                   section['lat_start'],
                                   section['lon_end'],
                                   section['lat_end'],
                                   1000
                                   )
        # bring into the desired shape [[],...,[]]
        section_waypoints = [[section_waypoints[i][0], section_waypoints[i][1]]
                             for i in range(len(section_waypoints))]

    else:
        # Compute the 'linear' connection between the section start and end
        section_lon = np.linspace(section['lon_start'],
                                  section['lon_end'],
                                  1000
                                  )

        section_lat = np.linspace(section['lat_start'],
                                  section['lat_end'],
                                  1000
                                  )

        # Bring the section coordinates into the disired shape [[],...,[]]
        section_waypoints = [[section_lon[i], section_lat[i]] for i in range(len(section_lat))]

    return section_waypoints, mesh, section


def _Haversine(lon1, lat1, lon2, lat2, use_great_circle):
    """
    havesine_np,py

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    https://gist.github.com/susanli2016/57f37514fbc491e287c300616104fe77
    In case the section is zonal, compute the distance along the latitude.

    All args must be of equal length.

    """
    if use_great_circle:
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c

    else:
        lat1, lat2 = map(np.radians, [lat1, lat2])
        dlon = np.abs(lon2 - lon1)
        km = 111.321 * np.cos(lat1) * dlon

    return km


def _ReduceMeshElementNumber(section_waypoints, mesh, section, add_extent, use_great_circle):
    '''
    reduce_element_number.py

    Reduces the number of elements that are loaded into memory to those nearby the section

    Inputs
    ------
    section_waypoints (list)
        list with all section waypoints [[lon,lat], ... ,[lon,lat]]
    mesh (fesom.mesh object)
        fesom.mesh
    section (dict)
        section dictionary
    add_extent (int, float)
        extent in degree that is added to the box cutout to contain all elements needed, choose small in case of high resolution meshes

    Returns
    -------
    elem_box_nods (list)
        list of indices that define the three nods of each element that belongs to the box
    elem_box_indices (list)
        list of indices where no_nan_triangles == True (to select the right elements when loading the data)
    '''

    # write section longitude and latitude in separate lists
    section_waypoints_lon = [section_waypoints[i][0] for i in range(len(section_waypoints))]
    section_waypoints_lat = [section_waypoints[i][1] for i in range(len(section_waypoints))]

    if add_extent < 1:
        warnings.warn(
            'The extend added to the box is small, this might lead to errors when using low resolution meshes')

    # find the maximum and minumum zonal/ meridional extent of the section
    box_mesh = [min(section_waypoints_lon) - add_extent,
                max(section_waypoints_lon) + add_extent,
                min(section_waypoints_lat) - add_extent,
                max(section_waypoints_lat) + add_extent
                ]

    # find the elements that are within the extent
    elem_no_nan, no_nan_triangles = cut_region(mesh, box_mesh)
    no_cyclic_elem2 = get_no_cyclic(mesh, elem_no_nan)
    elem_box_nods = elem_no_nan[no_cyclic_elem2]

    # create an array containing the indices of the elements that belong to the region
    elem_box_indices = np.arange(mesh.e2d)[no_nan_triangles]

    # Compute the distance of each section coodinate to the center of each element to further reduce the amount of polygons needed
    # in case of meridional or zonal section the chosen box is already small enough to be loaded and no further elements have to be removed
    # in all other cases the rectangular box gets to large and needs further shrinking
    if section['orientation'] == 'other':
        min_dist = add_extent * 100  # minimum distance in km to take element into account
        distance_bool = list()

        # compute the center of each element
        element_center_lon = np.mean(mesh.x2[elem_box_nods], axis=1)
        element_center_lat = np.mean(mesh.y2[elem_box_nods], axis=1)

        for ii in range(len(element_center_lat)):
            lon_temp = np.repeat(element_center_lon[ii], len(section_waypoints_lon))
            lat_temp = np.repeat(element_center_lat[ii], len(section_waypoints_lat))

            distances = _Haversine(lon_temp,
                                   lat_temp,
                                   section_waypoints_lon,
                                   section_waypoints_lat,
                                   False
                                   )

            if any(distances <= min_dist):
                distance_bool.append(True)
            else:
                distance_bool.append(False)

        # remove the elements that are to far away from the section
        elem_box_nods = elem_box_nods[distance_bool]
        elem_box_indices = elem_box_indices[distance_bool]

    return elem_box_nods, elem_box_indices


def _LinePolygonIntersections(mesh, section_waypoints, elem_box_nods, elem_box_indices):
    '''
    line_polygon_intersections.py

    Creates shapely polygon and line elements for the section and the mesh elements and computes the intersection coordinates

    Inputs
    ------
    mesh (fesom.mesh object)
        mesh object
    section_waypoints (list)
        list containing the waypoints
    elem_box_nods (list)
        list of indices that defines the three nods of each element that belongs to the box
    elem_box_indices (list)
        list of indices where no_nan_triangles == True (to select the right elements when loading the data)

    Returns
    -------
    '''
    # CREATE SHAPELY LINE AND POLYGON ELEMENTS
    line_section = sg.LineString(section_waypoints)

    polygon_list = list()

    for ii in range(elem_box_nods.shape[0]):
        polygon_list.append(
            sg.Polygon(
                [
                    (mesh.x2[elem_box_nods][ii, 0], mesh.y2[elem_box_nods][ii, 0]),
                    (mesh.x2[elem_box_nods][ii, 1], mesh.y2[elem_box_nods][ii, 1]),
                    (mesh.x2[elem_box_nods][ii, 2], mesh.y2[elem_box_nods][ii, 2]),
                ]
            )
        )

    ###################
    # Returns
    # line_section: shapely line element that contains the section coordinates
    # polygon_list: list of shapely polygons that contains all nearby elements
    ###################

    # COMPUTE THE INTERSECTION COORDINATES OF THE POLYGONS AND THE LINE ELEMENT
    intersection_bool = list()
    intersection_coords = list()
    intersection_points = list()

    # check for intersections
    for ii in range(len(polygon_list)):
        intersection = polygon_list[ii].intersection(line_section).coords

        # if no intersections (coords == [])
        if not intersection:
            intersection_bool.append(False)  # fill boolean array with False (no intersects)

        # if exist intersections (coords != [] )
        else:
            intersection_bool.append(True)  # fill boolean array with True (intersects exists)
            # fill the intersection coordinates list with the shapely intersection coordinates object
            intersection_coords.append(intersection)

    # remove all intersections that are not at the edge of the elements but inside (only first and last intersection coordinates are considered)
    cell_intersections = list()

    for intersection in intersection_coords:
        cell_intersections.append([(list(intersection)[0]), (list(intersection)[-1])])

    # remove indices of elements that are not intersected
    elem_box_nods = elem_box_nods[intersection_bool]
    elem_box_indices = elem_box_indices[intersection_bool]

    return elem_box_nods, elem_box_indices, cell_intersections


def _CreateVerticalGrid(cell_intersections, section, mesh, use_great_circle):
    '''
    vertical_grid.py

    Compute the properties of the vertical section grid

    Inputs
    ------
    cell_intersections (list)
        list of the edge intersection coordinates of each element in shapely.coordinates format
    section (dict)
        section dictionary
    mesh (fesom.mesh onject)
        fesom.mesh

    Returns
    -------
    distances_between (numpy.ndarray)
        horizontal distance that each element is intersected by the section
    distances_to_start (numpy.ndarray)
        absolute distance of the center of the intersection segment to the section start
    grid_cell_area (numpy.ndarray)
        vertical area of the intersected elements
    layer_thickness (numpy.ndarray)
        thickness of individual layers

    '''
    distances_between = []
    distances_to_start = []

    for ii in range(len(cell_intersections)):
        distances_between.append(_Haversine(cell_intersections[ii][0][0],  # lon1
                                            cell_intersections[ii][0][1],   # lat1
                                            cell_intersections[ii][1][0],   # lon2
                                            cell_intersections[ii][1][1],  # lat2
                                            use_great_circle
                                            )
                                 )

        distances_to_start.append(_Haversine(section['lon_start'],
                                             section['lat_start'],
                                             (cell_intersections[ii][0][0] +
                                              cell_intersections[ii][1][0]) / 2,
                                             (cell_intersections[ii][0][1] +
                                              cell_intersections[ii][1][1]) / 2,
                                              use_great_circle
                                             )
                                  )

    distances_between = np.array(distances_between) * 1000  # scale to m
    distances_to_start = np.array(distances_to_start) * 1000  # scale to m

    layer_thickness = abs(np.diff(mesh.zlev))  # vertical layer thickness
    grid_cell_area = distances_between[:, np.newaxis] * \
        layer_thickness[np.newaxis, :]  # area of the intersected elements

    return distances_between, distances_to_start, layer_thickness, grid_cell_area


def _CreateDataset(files, mesh, elem_box_indices, elem_box_nods, distances_between, distances_to_start, grid_cell_area, how, abg, chunks):
    '''
    create_dataset.py

    Load and unrotate the velocities on the elements that belong to the section and add variables to the dataset

    Inputs
    ------
    files (list)
        list of files to be loaded (u.fesom and v.fesom)
    elem_box_indices (list)
        list of indices that belong points towards the elements that belong to the section
    elem_box_nods (list)
        list of indices of the three nods that form each element
    distances_between (np.ndarray)
        list of the horizontal length of the single segments in m
    distances_to_start (np.ndarray)
        distance of the segment center to the starting point in m
    grid_cell_area (np.ndarray)
        cell weight for the transport calculations in m2
    how (str)
        either 'mean' or 'ori'
    abg (list)
        euler angles to rotate the fesom2 velocity output (default, [50, 15, -90])
    chunks (dict)
        chunks of the velocity dataset (default {'elem': 1e4})

    Returns
    -------
    ds (xarray.Dataset)
        dataset with the velocities

    '''

    # LOAD THE VELOCITY DATA
    if how == 'ori':
        ds = xr.open_mfdataset(files, combine='by_coords', chunks=chunks).isel(
            elem=elem_box_indices).load()
    elif how == 'mean':
        ds = xr.open_mfdataset(files, combine='by_coords', chunks=chunks).isel(
            elem=elem_box_indices).mean(dim='time').load()

    # rename u and v to u_rot, v_rot
    ds = ds.rename({'u': 'u_rot'})
    ds = ds.rename({'v': 'v_rot'})

    # ADD SOME FURTHER VARIABLES
    ds.assign_coords({'triple': ("triple", [1, 2, 3])})

    # elem_indices
    ds['elem_indices'] = (('elem'), elem_box_indices)
    ds.elem_indices.attrs['description'] = 'indices of the elements that belong to the section relative to the global data field'

    # elem_nods
    ds['elem_nods'] = (('elem', 'triple'), elem_box_nods)
    ds.elem_nods.attrs['description'] = 'indices of the 3 nods that represent the elements that belong to the section relative to the global data field'

    # horizontal_distance
    ds['horizontal_distance'] = (('elem'), distances_between)
    ds.horizontal_distance.attrs['description'] = 'width of the intersection for each element'
    ds.horizontal_distance.attrs['units'] = 'm'

    # distance_to_start
    ds['distances_to_start'] = (('elem'), distances_to_start)
    ds.distances_to_start.attrs['description'] = 'horizontal distance of the center of the segment to the start of the section'
    ds.distances_to_start.attrs['units'] = 'm'

    # vertical_cell_area
    ds['vertical_cell_area'] = (('elem', 'nz1'), grid_cell_area)
    ds.vertical_cell_area.attrs['description'] = 'cell area of the single intersected elements'
    ds.vertical_cell_area.attrs['units'] = 'm^2'

    # UNROTATE
    lon_elem_center = np.mean(mesh.x2[ds.elem_nods], axis=1)
    lat_elem_center = np.mean(mesh.y2[ds.elem_nods], axis=1)

    u, v = vec_rotate_r2g(abg[0], abg[1], abg[2], lon_elem_center[np.newaxis, :, np.newaxis],
                             lat_elem_center[np.newaxis, :, np.newaxis], ds.u_rot.values, ds.v_rot.values, flag=1)

    ds['u'] = (('time', 'elem', 'nz1'), u)
    ds['v'] = (('time', 'elem', 'nz1'), v)

    return ds


def _ComputeTransports(ds, mesh, section, cell_intersections, section_waypoints, use_great_circle):
    '''
    compute_transports.py

    Computes the transports across the section by taking the dot product of the section normal vector and the local velocity vector

    Inputs
    ------
    ds (xarray.Dataset)
        dataset containing the velocities etc.
    mesh (fesom.mesh object)
        fesom.mesh
    section (dict)
        section dictionary
    cell_intersections (list)
        list with all the two intersection coordinates of each mesh element
    section_waypoints (list)
        list of all the section waypoints
    use_great_circle (bool)
        True or False

    Returns
    -------
    ds (xarray.Dataset)
        final dataset

    '''

    # COMPUTE THE NORMAL VECTORS TO THE SECTION

    # in all section['orientation'] == 'other' cases the normal vector of the intersection segments have to be computed individually
    if section['orientation'] == 'other':
        # compute the normal vector for each section segment, segments are computed from west to east
        # in this case, the normal vector will always point towards north-east
        segment_vectors = np.ones((len(cell_intersections), 2))
        normal_vectors = np.ones((len(cell_intersections), 2))

        # compute the single segment vectors connecting the two intercections of each element (vec(AB) = B - A)
        # compute the normal vector for each of the segment vectors
        for ii in range(len(cell_intersections)):
            segment_vectors[ii, :] = np.array(
                cell_intersections[ii][1]) - np.array(cell_intersections[ii][0])

            # 2D normal vector for (a,b) is (-b,a)
            normal_vectors[ii, 0] = -segment_vectors[ii, 1]
            normal_vectors[ii, 1] = segment_vectors[ii, 0]

        # normalize normal vector
        norm = np.sqrt(np.sum(normal_vectors**2, axis=1))**(-1)
        normal_vectors = normal_vectors * norm[:, np.newaxis]

        # length test: norm of normal vector has to == 1 (|n| = 1)
        length_test = np.sqrt(np.sum(normal_vectors**2, axis=1))
        if any(1 - np.abs(length_test) > 1e-10):
            raise ValueError('Length of the normalized normal vector != 1 +- 1e-10')

        # angle test: angle between segment and normal vector == 0 (dot product == 0)
        angle_test = [np.dot(segment_vectors[i, :], normal_vectors[i, :])
                      for i in range(len(segment_vectors))]
        if any(np.abs(angle_test) > 1e-5):
            raise ValueError('Angle between normalized normal vector and segment vector != 90°')

    # COMPUTE TRANSPORT ACROSS SECTION

    # in meridional case the section is a great circle anyway and the across section velocity is given by u
    if section['orientation'] == 'meridional':
        # * ds.vertical_cell_area.values[np.newaxis,:,:])
        ds['velocity_across'] = (('time', 'elem', 'nz1'), ds.u.values)

    # in zonal case with no great circle the across section velocity is given by v
    elif (section['orientation'] == 'zonal') & (use_great_circle == False):
        # * ds.vertical_cell_area.values[np.newaxis,:,:])
        ds['velocity_across'] = (('time', 'elem', 'nz1'), ds.v.values)

    # in all other cases the across section velocity is the dot product of velocity and section normal vector
    elif section['orientation'] == 'other':
        # split the normal vector into x and y part
        normal_x = normal_vectors[:, 0]
        normal_y = normal_vectors[:, 1]

        ds['velocity_across'] = ds.u * normal_x[np.newaxis, :, np.newaxis] + \
            ds.v * normal_y[np.newaxis, :, np.newaxis]

    # compute transport across section
    ds['transport_across'] = (('time', 'elem', 'nz1'), ds['velocity_across'].values *
                              ds.vertical_cell_area.values[np.newaxis, :, :])

    # add attributes
    ds.transport_across.attrs['description'] = 'volume transport of each single cell through the section'
    ds.transport_across.attrs['units'] = 'm^3/s'

    ds.velocity_across.attrs['description'] = 'across section velocity'
    ds.velocity_across.attrs['units'] = 'm/s'

    # Drop unwanted VARIABLES
    ds = ds.drop(['u_rot', 'v_rot'])

    # SORTBY DISTANCE
    ds = ds.sortby('distances_to_start')

    return ds


def _AddTempSalt(section, ds, data_path, mesh):
    '''
    _AddTempSalt.py

    Adds temperature and salinity values to the section. The temperature and salinity is converted from nods to elements by taking the average
    of the three nods that form the element.

    Inputs
    ------
    section (dict)
        section dictionary
    ds (xarray.Dataset)
        dataset containing the velocities etc.
    data_path (str)
        directory where the fesom output is stored
    mesh (fesom mesh file)
        fesom mesh file

    Returns
    -------

    ds (xr.Dataset)
        final dataset


    '''

    # Check for existance of the files
    years = section['years']
    files_temp = [data_path + 'temp.fesom.' + str(year) + '.nc' for year in years]
    files_salt = [data_path + 'salt.fesom.' + str(year) + '.nc' for year in years]
    files = files_temp + files_salt

    file_check = []
    for file in files:
        file_check.append(isfile(file))

    if not all(file_check):
        raise FileExistsError('One or more of the temperature/ salinity files do not exist!')

    # Open files
    ds_ts = xr.open_mfdataset(files, combine='by_coords', chunks={'nod2': 1e4})

    # Only load the nods that belong to elements that are part of the section
    # Flatten the triplets first
    ds_ts = ds_ts.isel(nod2=ds.elem_nods.values.flatten()).load()

    # Reshape to triplets again and average all three values to obtain an estimate of the elements properties
    temp = ds_ts.temp.values.reshape(len(ds.time), len(ds.elem_nods), 3, mesh.nlev - 1).mean(axis=2)
    salt = ds_ts.salt.values.reshape(len(ds.time), len(ds.elem_nods), 3, mesh.nlev - 1).mean(axis=2)

    # Add to dataset
    ds['temp'] = (('time', 'elem', 'nz1'), temp)
    ds['salt'] = (('time', 'elem', 'nz1'), salt)

    return ds



def _AddIceTransport(section, ds, data_path, mesh, abg):

    '''
    _AddIceTransport.py

    Adds the ice volume transport across the section by averaging the ice velocity to of 3 nods of the intersected elements.

    Inputs
    ------
    section (dict)
        section dictionary
    ds (xarray.Dataset)
        dataset containing the velocities etc.
    data_path (str)
        directory where the fesom output is stored
    mesh (fesom mesh file)
        fesom mesh file

    Returns
    -------

    ds (xr.Dataset)
        final dataset

    '''

    # Check for existance of the files
    years = section['years']
    files_uice = [data_path + 'uice.fesom.' + str(year) + '.nc' for year in years]
    files_vice = [data_path + 'vice.fesom.' + str(year) + '.nc' for year in years]
    files_mice = [data_path + 'm_ice.fesom.' + str(year) + '.nc' for year in years]

    files = files_uice + files_vice + files_mice

    file_check = []
    for file in files:
        file_check.append(isfile(file))

    if not all(file_check):
        raise FileExistsError('One or more of the ice velocity files do not exist!')

    # prohibit the orientation=other case
    if section['orientation'] == 'other':
        warnings.warn('Currently the ice transport across non-zonal/ non-meridional sections is not implemented! \
                        Skipping ice transport computation...')
    else:
        # Open files
        ds_ice = xr.open_mfdataset(files, combine='by_coords')

        # Only load the nods that belong to elements that are part of the section
        # Flatten the triplets first
        ds_ice_section = ds_ice.isel(nod2=ds.elem_nods.values.flatten())

        # Reshape to triplets again
        m_ice_nods = ds_ice_section.m_ice.values.reshape((len(ds_ice_section.time), len(ds.elem), 3))
        u_ice_nods = ds_ice_section.uice.values.reshape((len(ds_ice_section.time), len(ds.elem), 3))
        v_ice_nods = ds_ice_section.vice.values.reshape((len(ds_ice_section.time), len(ds.elem), 3))

        # Rotate the velocity vectors
        lon_elem_center = np.mean(mesh.x2[ds.elem_nods], axis=1)
        lat_elem_center = np.mean(mesh.y2[ds.elem_nods], axis=1)

        u_ice_nods, v_ice_nods = vec_rotate_r2g(abg[0], abg[1], abg[2], lon_elem_center[np.newaxis, :, np.newaxis],
                                                   lat_elem_center[np.newaxis, :, np.newaxis], u_ice_nods, v_ice_nods, flag=1)

        # Write the triplets to the dataset
        ds['m_ice_nods'] = (('time','elem','tri'), m_ice_nods)
        ds['u_ice_nods'] = (('time','elem','tri'), u_ice_nods)
        ds['v_ice_nods'] = (('time','elem','tri'), v_ice_nods)

        # Average the triplets and write to dataset
        ds['m_ice'] = ds.m_ice_nods.mean(dim='tri')
        ds['u_ice'] = ds.u_ice_nods.mean(dim='tri')
        ds['v_ice'] = ds.v_ice_nods.mean(dim='tri')

        # Compute the across section ice transport in m^3/s
        if section['orientation'] == 'zonal':
            ds['ice_transport_across'] = ds.horizontal_distance * ds.m_ice * ds.v_ice

        elif section['orientation'] == 'meridional':
            ds['ice_transport_across'] = ds.horizontal_distance * ds.m_ice * ds.u_ice

    return ds


def cross_section_transports(section,
                             mesh_path,
                             data_path,
                             mesh_diag_path,
                             years,
                             use_great_circle=True,
                             how='mean',
                             add_extent=1,
                             abg=[50, 15, -90],
                             add_TS=False,
                             add_IT=False,
                             chunks={'elem': 1e4}

                             ):
    '''
    cross_section_transports.py

    Computes the horizontal transport across a vertical section from fesom2 velocity output on mesh elements, by computing the intersections
    of the section with the elements.

    Inputs
    ------
    section (list, str)
        either a list of the form [lon_start, lon_end, lat_start, lat_end] or a string for a preset section: 'FRAMSTRAIT', 'BSO'
    mesh_path (str)
        directory where the mesh files are stored
    data_path (str)
        directory where the data is stored
    mesh_diag_path (str: optional, default=None)
        directory where the mesh_diag file is stored, if None it is assumed to be located in data_path
    use_great_circle (bool)
        compute the section waypoints along a great great circle (default=True)
    how (str)
        either 'mean' for time mean transport or 'ori' for original data (default='mean')
    add_extent (int, float)
        the additional extent of the cutoutbox [lon_start, lon_end, lat_start, lat_end],
        choose as small as possible (small for high resolution meshes and large for low resolution meshes)
        this will impove the speed of the function (default = 1°)
    abg (list)
        rotation of the velocity data (default=[50,15,-90])
    add_TS (bool)
        add temperature and salinity to the section (default=False)
    add_IT (bool)
        add ice transport across the section (default=False)
    chunks (dict)
        chunks for parallelising the velocity data (default: chunks={'elem': 1e4})

    Returns
    -------
    ds (xarray.Dataset)
        dataset containing all output variables
    section (dict)
        dictionary containing all section information

    '''

    # Wrap the subfunctions up
    mesh, mesh_diag, files, section = _ProcessInputs(
        section, mesh_path, data_path, mesh_diag_path, years, how, use_great_circle)

    section_waypoints, mesh, section = _ComputeWaypoints(
        section, mesh, use_great_circle)

    elem_box_nods, elem_box_indices = _ReduceMeshElementNumber(
        section_waypoints, mesh, section, add_extent, use_great_circle)

    elem_box_nods, elem_box_indices, cell_intersections = _LinePolygonIntersections(
        mesh, section_waypoints, elem_box_nods, elem_box_indices)

    distances_between, distances_to_start, layer_thickness, grid_cell_area = _CreateVerticalGrid(
        cell_intersections, section, mesh, use_great_circle)

    ds = _CreateDataset(files, mesh, elem_box_indices, elem_box_nods,
                        distances_between, distances_to_start, grid_cell_area, how, abg, chunks)

    ds = _ComputeTransports(ds, mesh, section, cell_intersections,
                            section_waypoints, use_great_circle)

    if add_TS:
        ds = _AddTempSalt(section, ds, data_path, mesh)

    if add_IT:
        ds = _AddIceTransport(section, ds, data_path, mesh, abg)

    return ds, section
