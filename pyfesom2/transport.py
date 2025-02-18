"""
Module for computing transports across sections from fesom2 output
Author: Finn Heukamp (finn.heukamp@awi.de)
Initial version: 23.11.2021
"""

import warnings
from os.path import isfile, isdir
import xarray as xr
import numpy as np
import shapely.geometry as sg
import pyproj
import pymap3d as pm
from dask.diagnostics import ProgressBar
from tqdm.notebook import tqdm
from .load_mesh_data import load_mesh
from .ut import vec_rotate_r2g, get_no_cyclic, cut_region


def _ProcessInputs(section, data_path, years, n_points):
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
    years (np.ndarray)
        years to compute
    mesh_diag_path (str: optional, default=None)
        directory where the mesh_diag file is stored, if None it is assumed to be located in data_path
    n_points (int)
        number of waypoints between start and end of section

    Returns
    -------
    mesh (fesom.mesh object)
        fesom mesh
    mesh_diag (xr.dataset)
        fesom mesh diag
    section (dict)
        section dictionary containing additional information
    files (list)
        list of velocity files


        '''

    print('Starting computation...')

    # Check the input data types
    if not isinstance(section, list) | isinstance(section, str):
        raise ValueError(
            'The section must be a list of form [lon_start, lon_end, lat_start, lat_end] or a string for a preset section ("FS", "BSO", "BSX", ...)')

    if isinstance(section, list) & (len(section) != 4):
        raise ValueError(
            'The section must be a list of form [lon_start, lon_end, lat_start, lat_end]')

    if not isinstance(n_points, int):
        raise ValueError(
            'n_points must be an integer!'
        )

    # Check for existance of the files
    files_u = [data_path + 'u.fesom.' + str(year) + '.nc' for year in years]
    files_v = [data_path + 'v.fesom.' + str(year) + '.nc' for year in years]

    files = files_u + files_v

    file_check = []
    for file in files:
        file_check.append(isfile(file))

    if not all(file_check):
        raise FileExistsError('One or more of the velocity files do not exist!')

    return files

def _CreateLoadSection(section):
    '''
    Load the section parameters from present or create from custom section_name

    Inputs
    ------
    section (list, str)
        either a list of the form [lon_start, lon_end, lat_start, lat_end] or a string for a preset section

    Returns
    -------
    section (dict)
        section dictionary
    '''

    # Create the section dictionary from preset
    if isinstance(section, str):
        section_name = section

        presets = ["BSO", "BSX", "ST_ANNA_TROUGH", "FRAMSTRAIT", "FRAMSTRAIT_FULL",
                  "BSO_FULL", "BS_40E"]

        if not section_name in presets:
            raise ValueError('The chosen preset section does not exist! Choose from:' + str(presets)
                                + ' or add your own preset to _CreateLoadSection.py in pyfesom.transport.py')
        else:
            if section_name == 'BSO':
                section = {'lon_start': 19.999,
                           'lon_end': 19.999,
                           'lat_start': 74.5,
                           'lat_end': 70.08,
                           }
            if section_name == 'BSO_FULL':
                section = {'lon_start': 19.999,
                           'lon_end': 19.999,
                           'lat_start': 78.8,
                           'lat_end': 70.08,
                           }

            elif section_name == 'BSX':
                section = {'lon_start': 64,
                           'lon_end': 64,
                           'lat_start': 76,
                           'lat_end': 80.66,
                           }

            elif section_name == 'FRAMSTRAIT_FULL':
                section = {'lon_start': -18.3,#-6,
                           'lon_end': 10.6,
                           'lat_start': 78.8,
                           'lat_end': 78.8,
                           }

            elif section_name == 'FRAMSTRAIT':
                section = {'lon_start': -6,
                           'lon_end': 10.6,
                           'lat_start': 78.8,
                           'lat_end': 78.8,
                           }

            elif section_name == 'ST_ANNA_TROUGH':
                section = {'lon_start': 60,
                           'lon_end': 80,
                           'lat_start': 80,
                           'lat_end': 80,
                           }

            elif section_name == 'BS_40E':
                section = {'lon_start': 40,
                           'lon_end': 40,
                           'lat_start': 68,
                           'lat_end': 80,
                           }

            # add more presets here

        section['name'] = section_name

    # create custom section dict
    elif isinstance(section, list):
        section = {'lon_start': section[0],
                   'lon_end': section[1],
                   'lat_start': section[2],
                   'lat_end': section[3],
                   }
        section['name'] = 'not specified'

    # Find the orientation of the section and look for the nesseccary velocity files
    if section['lon_start'] == section['lon_end']:
        section['orientation'] = 'meridional'

    elif (section['lat_start'] == section['lat_end']) :
        section['orientation'] = 'zonal'

    else:
        section['orientation'] = 'other'
        raise ValueError('Only zonal or meridional are currently supported!')

    print('\nYour section: ', section['name'], ': Start: ', section['lon_start'], '°E ', section['lat_start'], '°N ', 'End: ', section['lon_end'], '°E ', section['lat_end'], '°N')

    return section

def _ComputeWaypoints(section, mesh, use_great_circle, n_points):
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


    Returns
    -------
    section waypoints ()
        waypoints along the section
    mesh (fesom.mesh object)
        fesom mesh
    section (dict)
        dictionary containing section information
    '''

    if use_great_circle:
        # Compute the great circle coordinates along the section
        g = pyproj.Geod(ellps='WGS84')

        section_waypoints = g.npts(section['lon_start'],
                                   section['lat_start'],
                                   section['lon_end'],
                                   section['lat_end'],
                                   n_points
                                   )
        # bring into the desired shape [[],...,[]]
        section_waypoints = [[section_waypoints[i][0], section_waypoints[i][1]]
                             for i in range(len(section_waypoints))]

    else:
        # Compute the 'linear' connection between the section start and end
        section_lon = np.linspace(section['lon_start'],
                                  section['lon_end'],
                                  n_points
                                  )

        section_lat = np.linspace(section['lat_start'],
                                  section['lat_end'],
                                  n_points
                                  )

        # Bring the section coordinates into the disired shape [[],...,[]]
        section_waypoints = [[section_lon[i], section_lat[i]] for i in range(len(section_lat))]

    return section_waypoints, mesh, section

def _ReduceMeshElementNumber(section_waypoints, mesh, section, add_extent):
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
    elem_box_indices = np.arange(mesh.e2d)[no_nan_triangles][no_cyclic_elem2]

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
    elem_box_nods (list)
        list of indices that define the three nods of each element that belongs to the box
    elem_box_indices (list)
        list of indices where no_nan_triangles == True (to select the right elements when loading the data)
    cell_intersections (list)
        list with all intersections between the line element and the polygons
    line_section (shapely.line)
        shapely line element that represents the section
    '''
    # CREATE SHAPELY LINE AND POLYGON ELEMENTS
    line_section = sg.LineString(section_waypoints)

    polygon_list = list()

    print('\nConverting grid cells to Polygons... (If this takes very long try to reduce the add_extent parameter)')
    for ii in tqdm(range(elem_box_nods.shape[0])):
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
    print('Looking for intersected grid cells...')
    for ii in tqdm(range(len(polygon_list))):
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
        cell_intersections.append([[list(intersection)[0]], [list(intersection)[-1]]])

    # remove indices of elements that are not intersected
    elem_box_nods = elem_box_nods[intersection_bool]
    elem_box_indices = elem_box_indices[intersection_bool]

    return elem_box_nods, elem_box_indices, cell_intersections, line_section

def _FindIntersectedEdges(mesh, elem_box_nods, elem_box_indices, line_section, cell_intersections):
    '''
    Find the two intersected edges of each mesh element along the section (2 out of three). In case the start/ end point is in the ocean only one edge is
    intersected. In this case the associated mesh element is dropped.

    Inputs
    ------
    mesh (fesom.mesh object)
        mesh object
    elem_box_nods (list)
        list of indices that defines the three nods of each element that belongs to the box
    elem_box_indices (list)
        list of indices where no_nan_triangles == True (to select the right elements when loading the data)
    cell_intersections (list)
        list with all intersections between the line element and the polygons
    line_section (shapely.line)
        shapely line element that represents the section

    Returns
    -------
    intersected_edge (np.ndarray)
        boolean array, True if edge of element is intersected, False otherwise
    midpoints_edge (np.ndarray)
        centers of the three edges asociated to each single mesh element
    elem_centers (np.ndarray)
        cener of the mesh element
    elem_box_nods (list)
        list of indices that defines the three nods of each element that belongs to the box
    elem_box_indices (list)
        list of indices where no_nan_triangles == True (to select the right elements when loading the data)
    cell_intersections (list)
        list with all intersections between the line element and the polygons

    '''
    # array with the lons and lats of the three nods forming one element
    lon_elems = mesh.x2[elem_box_nods]
    lat_elems = mesh.y2[elem_box_nods]

    # array with the centers of the cells
    lon_centers, lat_centers = np.mean(mesh.x2[elem_box_nods], axis=1), np.mean(mesh.y2[elem_box_nods], axis=1)
    elem_centers = np.array([[lon_centers[i], lat_centers[i]] for i in range(len(lon_centers))])

    # Find the element edges that are intersected (2 of 3 regular, 1 of 3 with land)
    intersected_edge = np.ones((len(elem_centers),3), dtype=bool)
    midpoints_edge = np.zeros((len(elem_centers),3,2)) # elem, edge, (lon,lat)

    # iterate over all intersected elements
    for ii in range(len(elem_centers)):
        # extract the coordinates of the nods forming one element
        lon1, lon2, lon3 = lon_elems[ii][0], lon_elems[ii][1], lon_elems[ii][2]
        lat1, lat2, lat3 = lat_elems[ii][0], lat_elems[ii][1], lat_elems[ii][2]

        # compute the midpoints of the element edge
        midpoints_edge[ii,0,0] = (lon1+lon2)/2
        midpoints_edge[ii,1,0] = (lon2+lon3)/2
        midpoints_edge[ii,2,0] = (lon3+lon1)/2

        midpoints_edge[ii,0,1] = (lat1+lat2)/2
        midpoints_edge[ii,1,1] = (lat2+lat3)/2
        midpoints_edge[ii,2,1] = (lat3+lat1)/2

        # create shapely line elements for each of the element edges
        line12 = sg.LineString([[lon1,lat1], [lon2,lat2]])
        line23 = sg.LineString([[lon2,lat2], [lon3,lat3]])
        line31 = sg.LineString([[lon3,lat3], [lon1,lat1]])

        # find the element edges that intersect with the section
        if not list(line12.intersection(line_section).coords):
            intersected_edge[ii,0] = False
        if not list(line23.intersection(line_section).coords):
            intersected_edge[ii,1] = False
        if not list(line31.intersection(line_section).coords):
            intersected_edge[ii,2] = False

        # when there is only one edge of the element hit then set all intersections to False and drop it later
        if sum(intersected_edge[ii,:]) == 1:
            intersected_edge[ii,:] = False

    zeros_in_intersected_edge = np.where(intersected_edge.sum(axis=1) == 0)[0]
    #if len(zeros_in_intersected_edge) == 2:
    #    print('The section starts and ends in the ocean. Those elements that contain the start and end coordinate of the section are droped.')
    #elif len(zeros_in_intersected_edge) == 1:
    #    print('The section is land-ocean/ ocean-land. Those elements that contain the start and end coordinate of the section are droped.')
    #elif len(zeros_in_intersected_edge) == 0:
    #    print('The section is land to land')
    if len(zeros_in_intersected_edge) > 2:
        raise ValueError('Your section contains to many cell edges that were intersected only once. Only 0, 1 or 2 are allowed.')

    # Now drop those elements in the arrays
    elem_box_nods = np.delete(elem_box_nods, zeros_in_intersected_edge, axis=0)
    elem_box_indices = np.delete(elem_box_indices, zeros_in_intersected_edge)
    midpoints_edge = np.delete(midpoints_edge, zeros_in_intersected_edge, axis=0)
    elem_centers = np.delete(elem_centers, zeros_in_intersected_edge, axis=0)
    intersected_edge = np.delete(intersected_edge, zeros_in_intersected_edge, axis=0)
    cell_intersections = np.delete(np.array(cell_intersections).squeeze(), zeros_in_intersected_edge, axis=0)

    return intersected_edge, midpoints_edge, elem_centers, elem_box_indices, elem_box_nods, cell_intersections

def _BringIntoAlongPathOrder(midpoints_edge, intersected_edge, elem_centers, section):
    '''
    Brings the mesh elements and segment vectors into an along-section order (eastwards/ northwards).

    Inputs
    ------
    intersected_edge (np.ndarray)
        boolean array, True if edge of element is intersected, False otherwise
    midpoints_edge (np.ndarray)
        centers of the three edges asociated to each single mesh element
    elem_centers (np.ndarray)
        cener of the mesh element
    section (dict)
        section dictionary

    Returns
    -------
    c_lon (list)
        center longitude of mesh element
    c_lat (list)
        center latitude of mesh element
    f_lon (list)
        first edge midpoint latitude of the element
    f_lat (list)
        first edge midpoint longitude of the element
    s_lon (list)
         second edge midpoint latitude of the element
    s_lat (list)
         second edge midpoint longitude of the element
    elem_order (list)
        indices of the ascending elements


    '''
   #### FIND THE FIRST POINT OF THE SECTION

    if section['orientation'] == 'zonal':
        # find the westernnmost intersected edge midpoint
        start_ind = np.argmin(midpoints_edge[intersected_edge,0])
        start_value = midpoints_edge[intersected_edge,0][start_ind]

        # create list for already used elements
        first_element = list()

        for ii in range(midpoints_edge.shape[0]):

            # for each single midpoint tuple, check if the longitude is the same (then this is the first element)
            if start_value in midpoints_edge[ii,intersected_edge[ii,:],0]:
                first_element.append(ii)
                #print(first_element)

        #if len(first_element) > 1:
            #raise ValueError('Something is wrong here...')

        # now look which of the two intersected midpoints of the first element is intersected first
        ind_first = np.where(midpoints_edge[first_element[0],intersected_edge[first_element[0],:],0] == start_value)[0]

        # write the coordinates in the right order into lists (first_value, centeroid, second_value, (lon,lat)) for each element
        f_lon, f_lat, s_lon, s_lat, c_lon, c_lat, elem_order = list(), list(), list(), list(), list(), list(), list()

        c_lon.append(elem_centers[first_element[0],0])
        c_lat.append(elem_centers[first_element[0],1])
        f_lon.append(midpoints_edge[first_element[0], intersected_edge[first_element[0],:], 0][ind_first][0])
        f_lat.append(midpoints_edge[first_element[0], intersected_edge[first_element[0],:], 1][ind_first][0])
        s_lon.append(midpoints_edge[first_element[0], intersected_edge[first_element[0],:], 0][ind_first-1][0])# if ind_first =0 --> -1 which is the same index as 1
        s_lat.append(midpoints_edge[first_element[0], intersected_edge[first_element[0],:], 1][ind_first-1][0])

        elem_order.append(first_element[0])


        ###### Bring all the elements into the right order

        for jj in range(elem_centers.shape[0]-1):
            # Now we repeat this procedure for the second value of the previous element
            matching_element = list()
            for ii in range(midpoints_edge.shape[0]):
                # for each single midpoint tuple, check if the longitude is the same (then this is the next element)
                if s_lon[-1] in midpoints_edge[ii,intersected_edge[ii,:],0]:
                    matching_element.append(ii)
            #print(jj, matching_element)

            # apply some tests, the matching element has to have len() == 2 and the previous element must also be contained
            if (len(matching_element) != 2) | (elem_order[-1] not in matching_element):
                raise ValueError('Either your section hit an island or your add_extent parameter was chosen too small! ' +
                                'Increase the add_extent parameter as it might be too small for your mesh resolution! ' +
                                'Otherwise, the last working gridcell was at: ' +
                                 str(c_lon[-1]) + '°E, ' + str(c_lat[-1]) + '°N. ' +
                                 'Please use this coordinate tuple as the new start or end of the section! '
                                 )

            # find the matching element that's not the previous one, this is the next one
            if elem_order[-1] == matching_element[0]:
                ind = 1
            else:
                ind = 0

            # now look which of the two intersected midpoints of the element is the same as the last second value
            ind_first = np.where(midpoints_edge[matching_element[ind],intersected_edge[matching_element[ind],:],0] == s_lon[-1])[0]

            # append to list in right order
            c_lon.append(elem_centers[matching_element[ind],0])
            c_lat.append(elem_centers[matching_element[ind],1])
            f_lon.append(midpoints_edge[matching_element[ind], intersected_edge[matching_element[ind],:], 0][ind_first][0])
            f_lat.append(midpoints_edge[matching_element[ind], intersected_edge[matching_element[ind],:], 1][ind_first][0])
            s_lon.append(midpoints_edge[matching_element[ind], intersected_edge[matching_element[ind],:], 0][ind_first-1][0])# if ind_first =0 --> -1 which is the same index as 1
            s_lat.append(midpoints_edge[matching_element[ind], intersected_edge[matching_element[ind],:], 1][ind_first-1][0])

            elem_order.append(matching_element[ind])

    elif section['orientation'] == 'meridional':

        # find the southernmost intersected edge midpoint
        start_ind = np.argmin(midpoints_edge[intersected_edge,1])
        start_value = midpoints_edge[intersected_edge,1][start_ind]
        #print(start_ind, start_value)

        # create list for already used elements
        first_element = list()

        for ii in range(midpoints_edge.shape[0]):

            # for each single midpoint tuple, check if the latitude is the same (then this is the first element)
            if start_value in midpoints_edge[ii,intersected_edge[ii,:],1]:
                first_element.append(ii)
                #print(first_element)

        #if len(first_element) > 1:
            #raise ValueError('Something is wrong here...')

        # now look which of the two intersected midpoints of the first element is intersected first
        ind_first = np.where(midpoints_edge[first_element[0],intersected_edge[first_element[0],:],1] == start_value)[0]

        # write the coordinates in the right order into lists (first_value, centeroid, second_value, (lon,lat)) for each element
        f_lon, f_lat, s_lon, s_lat, c_lon, c_lat, elem_order = list(), list(), list(), list(), list(), list(), list()

        c_lon.append(elem_centers[first_element[0],0])
        c_lat.append(elem_centers[first_element[0],1])
        f_lon.append(midpoints_edge[first_element[0], intersected_edge[first_element[0],:], 0][ind_first][0])
        f_lat.append(midpoints_edge[first_element[0], intersected_edge[first_element[0],:], 1][ind_first][0])
        s_lon.append(midpoints_edge[first_element[0], intersected_edge[first_element[0],:], 0][ind_first-1][0])# if ind_first =0 --> -1 which is the same index as 1
        s_lat.append(midpoints_edge[first_element[0], intersected_edge[first_element[0],:], 1][ind_first-1][0])

        elem_order.append(first_element[0])


        ###### Bring all the elements into the right order

        for jj in range(elem_centers.shape[0]-1):
            # Now we repeat this procedure for the second value of the previous element
            matching_element = list()
            for ii in range(midpoints_edge.shape[0]):
                # for each single midpoint tuple, check if the longitude is the same (then this is the next element)
                if s_lat[-1] in midpoints_edge[ii,intersected_edge[ii,:],1]:
                    matching_element.append(ii)
            #print(jj, matching_element)

            # apply some tests, the matching element has to have len() == 2 and the previous element must also be contained
            if (len(matching_element) != 2) | (elem_order[-1] not in matching_element):
                raise ValueError('Either your section hit an island or your add_extent parameter was chosen too small! ' +
                                'Increase the add_extent parameter as it might be too small for your mesh resolution! ' +
                                'Otherwise, the last working gridcell was at: ' +
                                 str(c_lon[-1]) + '°E, ' + str(c_lat[-1]) + '°N. ' +
                                 'Please use this coordinate tuple as the new start or end of the section! '
                                 )

            # find the matching element that's not the previous one, this is the next one
            if elem_order[-1] == matching_element[0]:
                ind = 1
            else:
                ind = 0

            # now look which of the two intersected midpoints of the element is the same as the last second value
            ind_first = np.where(midpoints_edge[matching_element[ind],intersected_edge[matching_element[ind],:],0] == s_lon[-1])[0]

            # append to list in right order
            c_lon.append(elem_centers[matching_element[ind],0])
            c_lat.append(elem_centers[matching_element[ind],1])
            f_lon.append(midpoints_edge[matching_element[ind], intersected_edge[matching_element[ind],:], 0][ind_first][0])
            f_lat.append(midpoints_edge[matching_element[ind], intersected_edge[matching_element[ind],:], 1][ind_first][0])
            s_lon.append(midpoints_edge[matching_element[ind], intersected_edge[matching_element[ind],:], 0][ind_first-1][0])# if ind_first =0 --> -1 which is the same index as 1
            s_lat.append(midpoints_edge[matching_element[ind], intersected_edge[matching_element[ind],:], 1][ind_first-1][0])

            elem_order.append(matching_element[ind])

    #check if the no element appears twice
    for i in elem_order:
        if elem_order.count(i) > 1:
            raise ValueError('An element appeared twice while sorting...' + str(i))
    if len(elem_order) != elem_centers.shape[0]:
        raise ValueError('Wrong number of elements while sorting along path...')

    # Add the segments to the section dictionary
    section['f_lon'] = f_lon
    section['c_lon'] = c_lon
    section['s_lon'] = s_lon
    section['f_lat'] = f_lat
    section['c_lat'] = c_lat
    section['s_lat'] = s_lat


    return c_lon, c_lat, f_lon, f_lat, s_lon, s_lat, elem_order

def _ComputeBrokenLineSegments(f_lat, f_lon, s_lat, s_lon, c_lat, c_lon, section):
    '''
    Compute the two broken line segments that connect the intersected edge midpoints to the center of the mesh element
    in local cartesian coordinates. Afterwards compute the effective length of the two segments in x and y direction.

    Inputs
    ------
    c_lon (list)
        center longitude of mesh element
    c_lat (list)
        center latitude of mesh element
    f_lon (list)
        first edge midpoint latitude of the element
    f_lat (list)
        first edge midpoint longitude of the element
    s_lon (list)
         second edge midpoint latitude of the element
    s_lat (list)
         second edge midpoint longitude of the element
    section (dict)
        section dictionary

    Returns
    -------
    effective_dx (np.ndarray)
        the effective length of the two segment elements in x direction to compute transport with v
    effective_dy (np.ndarray)
        the effective length of the two segment elements in y direction to compute transport with u
    '''

    # create an array for the segment vectors (2 for each element) with the shape (elem, (dlon,dlat))
    first_segment_vector = np.ones((len(f_lon), 2))
    second_segment_vector = np.ones_like(first_segment_vector)
    for ii in range(len(f_lon)):

        # FIRST VECTOR OF THE ELEMENT
        # switch to a local cartesian coordinate system (centered at the element midpoint) and compute the vector connecting
        # the center of the intersected edge with the center of the element (always pointing outwards from the center of the element)
        dx, dy, dz = pm.geodetic2enu(lat0=c_lat[ii],
                                        lon0=c_lon[ii],
                                        h0=0,
                                        lat=f_lat[ii],
                                        lon=f_lon[ii],
                                        h=0,
                                     ell=pm.Ellipsoid.from_name('wgs84')
                                   )

        first_segment_vector[ii,0], first_segment_vector[ii,1] = -dx, -dy # turn the vector to point towards the center, in the direction of the section

        # SECOND VECTOR OF THE ELEMENT
        dx, dy, dz = pm.geodetic2enu(lat0=c_lat[ii],
                                        lon0=c_lon[ii],
                                        h0=0,
                                        lat=s_lat[ii],
                                        lon=s_lon[ii],
                                        h=0,
                                     ell=pm.Ellipsoid.from_name('wgs84')
                                   )

        second_segment_vector[ii,0], second_segment_vector[ii,1] = dx, dy

    # define the sign of the segment length
    if section['orientation'] == 'zonal':

        effective_dx = first_segment_vector[:,0] + second_segment_vector[:,0]
        effective_dy = -first_segment_vector[:,1] - second_segment_vector[:,1]


    if section['orientation'] == 'meridional':

        effective_dx = -first_segment_vector[:,0] - second_segment_vector[:,0]
        effective_dy = first_segment_vector[:,1] + second_segment_vector[:,1]


    return effective_dx, effective_dy

def _CreateVerticalGrid(effective_dx, effective_dy, mesh_diag):
    '''
    Creates the vertical grid to compute transports through the section

    Inputs
    ------
    effective_dx (np.ndarray)
        the effective length of the two segment elements in x direction to compute transport with v
    effective_dy (np.ndarray)
        the effective length of the two segment elements in y direction to compute transport with u

    Returns
    -------
    vertical_cell_area_dx (np.ndarray)
        the cell area for each mesh element to be multiplied by the meridional velocity
    vertical_cell_area_dy (np.ndarray)
        the cell area for each mesh element to be multiplied by the zonal velocity

    '''
    # take the layer thickness
    # old mesh_diag: zbar, new mesh_diag: nz
    try:
        layer_thickness = np.abs(np.diff(mesh_diag.zbar))
    except:
        layer_thickness = np.abs(np.diff(mesh_diag.nz))

    # compute the vertical area for dx and dy
    vertical_cell_area_dx = layer_thickness[:,np.newaxis] * effective_dx[np.newaxis,:]
    vertical_cell_area_dy = layer_thickness[:,np.newaxis] * effective_dy[np.newaxis,:]

    return vertical_cell_area_dx, vertical_cell_area_dy

def _AddMetaData(ds, elem_box_indices, elem_box_nods, effective_dx, effective_dy, vertical_cell_area_dx, vertical_cell_area_dy, c_lon, c_lat):
    '''
    Add some meta-data to the dataset.
    '''

     # ADD SOME FURTHER VARIABLES
    ds.assign_coords({'triple': ("triple", [1, 2, 3])})

    # elem_indices
    ds['elem_indices'] = (('elem'), elem_box_indices)
    ds.elem_indices.attrs['description'] = 'indices of the elements that belong to the section relative to the global data field'

    # elem_nods
    ds['elem_nods'] = (('elem', 'triple'), elem_box_nods)
    ds.elem_nods.attrs['description'] = 'indices of the 3 nods that represent the elements that belong to the section relative to the global data field'

    # horizontal_distances
    ds['zonal_distance'] = (('elem'), effective_dx)
    ds.zonal_distance.attrs['description'] = 'width of the two broken lines in each element in west-east direction'
    ds.zonal_distance.attrs['units'] = 'm'
    ds['meridional_distance'] = (('elem'), effective_dy)
    ds.meridional_distance.attrs['description'] = 'width of the two broken lines in each element in south-east direction'
    ds.meridional_distance.attrs['units'] = 'm'

    # vertical_cell_area
    ds['vertical_cell_area_dx'] = (('elem', 'nz1'), np.transpose(vertical_cell_area_dx))
    ds.vertical_cell_area_dx.attrs['description'] = 'cell area of the single intersected elements in east-west direction'
    ds.vertical_cell_area_dx.attrs['units'] = 'm^2'

    ds['vertical_cell_area_dy'] = (('elem', 'nz1'), np.transpose(vertical_cell_area_dy))
    ds.vertical_cell_area_dy.attrs['description'] = 'cell area of the single intersected elements in south-north direction'
    ds.vertical_cell_area_dy.attrs['units'] = 'm^2'

    # lon lat
    ds['lon_center'] = (('elem'), c_lon)
    ds.lon_center.attrs['description'] = 'longitude of the element centers'
    ds.lon_center.attrs['units'] = '°E'

    ds['lat_center'] = (('elem'), c_lat)
    ds.lat_center.attrs['description'] = 'latitude of the element centers'
    ds.lat_center.attrs['units'] = '°E'

    return ds

def _UnrotateLoadVelocity(how, files, elem_box_indices, elem_box_nods, vertical_cell_area_dx, vertical_cell_area_dy, c_lon, c_lat, effective_dx, effective_dy, elem_order, chunks, mesh, abg, uvrotated):
    '''
    Load and unrotate the fesom velocity files. Additionally bring the mesh elements into the right order (according to the section)

    Inputs
    ------
    how (str)
        mean or ori
    files (list)
        list of strings contianing the files to load
    elem_box_nods (list)
        list of indices that defines the three nods of each element that belongs to the box
    elem_box_indices (list)
        list of indices where no_nan_triangles == True (to select the right elements when loading the data)
    vertical_cell_area_dx (np.ndarray)
        the cell area for each mesh element to be multiplied by the meridional velocity
    vertical_cell_area_dy (np.ndarray)
        the cell area for each mesh element to be multiplied by the zonal velocity
    c_lon (list)
        center longitude of mesh element
    c_lat (list)
        center latitude of mesh element
    effective_dx (np.ndarray)
        the effective length of the two segment elements in x direction to compute transport with v
    effective_dy (np.ndarray)
        the effective length of the two segment elements in y direction to compute transport with u
    chunks (dict)
        chunks for dask (default: {'elem': 1e5}
    mesh (fesom.mesh object)
        fesom.mesh
    abg (list)
        mesh rotation [50 15 -90]
    uvrotated (bool)
        True: u, v are rotated; False: u, v are unrotated
        
    Returns
    -------
    ds (xr.Dataset)
        dataset containing all variables


    '''

    print('Loading the data into memory...')
     # decide on the loading strategy, for small datasets combine the data to one dataset, for large datasets load files individually
    overload = xr.open_dataset(files[0]).nbytes * 1e-9 * len(files) >= 25
    if overload:
        print('A lot of velocity data (' + str(np.round(xr.open_dataset(files[0]).nbytes * 1e-9 * len(files), decimals=2)) + 'GB)... This will take some time...')

    # Load and merge at the same time
    ProgressBar().register()
    ds = xr.open_mfdataset(files, combine='by_coords', chunks=chunks).isel(
        elem=elem_box_indices).load()

    ds = _AddMetaData(ds, elem_box_indices, elem_box_nods, effective_dx, effective_dy, vertical_cell_area_dx, vertical_cell_area_dy, c_lon, c_lat)

    if uvrotated:                             # only unrotate in case u, v are still rotated
        # rename u and v to u_rot, v_rot
        ds = ds.rename({'u': 'u_rot'})
        ds = ds.rename({'v': 'v_rot'})
        # UNROTATE
        lon_elem_center = np.mean(mesh.x2[ds.elem_nods], axis=1)
        lat_elem_center = np.mean(mesh.y2[ds.elem_nods], axis=1)
        try:
            u, v = vec_rotate_r2g(abg[0], abg[1], abg[2], lon_elem_center[np.newaxis, :, np.newaxis],
                                 lat_elem_center[np.newaxis, :, np.newaxis], ds.u_rot.values, ds.v_rot.values, flag=1)
        except:
            u, v = vec_rotate_r2g(abg[0], abg[1], abg[2], lon_elem_center[np.newaxis, :, np.newaxis],
                                 lat_elem_center[np.newaxis, :, np.newaxis], ds.u_rot.values.swapaxes(1,2), ds.v_rot.values.swapaxes(1,2), flag=1)
    
        ds['u'] = (('time', 'elem', 'nz1'), u)
        ds['v'] = (('time', 'elem', 'nz1'), v)
    
        ds = ds.drop_vars(['u_rot','v_rot'])

    # bring u and v into the right order
    ds['u'] = ds.u.isel(elem=elem_order)
    ds['v'] = ds.v.isel(elem=elem_order)

    if how == 'mean':
        ds = ds.mean(dim='time')

    return ds

def _TransportAcross(ds):
    '''
    Compute the transport across the broken line elements

    Inputs
    ------
    ds (xr.Dataset)
        dataset

    Returns
    -------
    ds (xr.Dataset)
        updated dataset


    '''
    ds['transport_across'] = ds.u * ds.vertical_cell_area_dy +  ds.v * ds.vertical_cell_area_dx

    return ds

def _AddTempSalt(section, ds, data_path, mesh, years, elem_order):
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
    files_temp = [data_path + 'temp.fesom.' + str(year) + '.nc' for year in years]
    files_salt = [data_path + 'salt.fesom.' + str(year) + '.nc' for year in years]
    files = files_temp + files_salt

    file_check = []
    for file in files:
        file_check.append(isfile(file))

    if not all(file_check):
        raise FileExistsError('One or more of the temperature/ salinity files do not exist!')

    overload = xr.open_dataset(files[0]).nbytes * 1e-9 * len(files) >= 25
    if overload:
        print('A lot of TS data (' + str(np.round(xr.open_dataset(files[0]).nbytes * 1e-9 * len(files), decimals=2)) + 'GB)... This will take some time...')

    # Open files
    ds_ts = xr.open_mfdataset(files, combine='by_coords', chunks={'nod2': 1e4})

    # Adjust dimension order
    ds_ts = ds_ts.transpose('time','nod2','nz1')
    
    # Only load the nods that belong to elements that are part of the section
    # Flatten the triplets first
    ds_ts = ds_ts.isel(nod2=ds.elem_nods.values.flatten()).load()

    # Reshape to triplets again and average all three values to obtain an estimate of the elements properties
    temp = ds_ts.temp.values.reshape(len(ds.time), len(ds.elem_nods), 3, mesh.nlev - 1).mean(axis=2)
    salt = ds_ts.salt.values.reshape(len(ds.time), len(ds.elem_nods), 3, mesh.nlev - 1).mean(axis=2)

    # Add to dataset
    ds['temp'] = (('time', 'elem', 'nz1'), temp)
    ds['salt'] = (('time', 'elem', 'nz1'), salt)

    # bring temp and sal into the right order
    ds['temp'] = ds.temp.isel(elem=elem_order)
    ds['salt'] = ds.salt.isel(elem=elem_order)

    return ds

def _OrderIndices(ds, elem_order):
    '''Brings the indices into the right order.

    Inputs
    ------
    ds (xr.dataset)
        dataset containing transport
    elem_order (list)
        order
    '''

    ds['elem_indices'] = ds.elem_indices.isel(elem=elem_order)
    ds['elem_nods'] = ds.elem_nods.isel(elem=elem_order)

    print('\n Done!')
    return ds

def _MaskBathymetry(ds, how):
    ''' Sets bathymetry values (gridcells where time mean transport is zero) to np.nan.

    Inputs
    ------
    ds (xr.dataset)
        dataset containing transport
    '''
    if how == 'ori':
        ds = ds.where(ds.transport_across.mean(dim='time') != 0, np.nan)
    elif how == 'mean':
        ds = ds.where(ds.transport_across != 0, np.nan)
        
    return ds

def cross_section_transport(section, mesh, data_path, years, mesh_diag, how='mean', add_extent=1, abg=[50, 15, -90], add_TS=False, chunks={'elem': 1e4}, use_great_circle=False, n_points=1000, uvrotated=True):
    '''
    Inputs
    ------
    section (list, str)
        either a list of the form [lon_start, lon_end, lat_start, lat_end] or a string for a preset section: 'FRAMSTRAIT', 'BSO'
    mesh (fesom.mesh file)
        fesom.mesh file
    data_path (str)
        directory where the data is stored
    mesh_diag (xr.Dataset)
        fesom.mesh.diag file
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
    chunks (dict)
        chunks for parallelising the velocity data (default: chunks={'elem': 1e4})
    n_points (int)
        number of waypoints between start and end of section
    uvrotated (bool)
        True: u, v FESOM output is in rotated coordinates; False: u, v FESOM output is already unroated (default=True)

    Returns
    -------
    ds (xarray.Dataset)
        dataset containing all output variables
    section (dict)
        dictionary containing all section information

    '''
    # Wrap up all the subroutines to a main function
    files = _ProcessInputs(section, data_path, years, n_points)

    section = _CreateLoadSection(section)

    section_waypoints, mesh, section = _ComputeWaypoints(section, mesh, use_great_circle, n_points)

    elem_box_nods, elem_box_indices = _ReduceMeshElementNumber(section_waypoints, mesh, section, add_extent)

    elem_box_nods, elem_box_indices, cell_intersections, line_section = _LinePolygonIntersections(mesh, section_waypoints, elem_box_nods, elem_box_indices)

    intersected_edge, midpoints_edge, elem_centers, elem_box_indices, elem_box_nods, cell_intersections = _FindIntersectedEdges(mesh, elem_box_nods, elem_box_indices, line_section, cell_intersections)

    c_lon, c_lat, f_lon, f_lat, s_lon, s_lat, elem_order = _BringIntoAlongPathOrder(midpoints_edge, intersected_edge, elem_centers, section)

    effective_dx, effective_dy = _ComputeBrokenLineSegments(f_lat, f_lon, s_lat, s_lon, c_lat, c_lon, section)

    vertical_cell_area_dx, vertical_cell_area_dy = _CreateVerticalGrid(effective_dx, effective_dy, mesh_diag)

    ds = _UnrotateLoadVelocity(how, files, elem_box_indices, elem_box_nods, vertical_cell_area_dx, vertical_cell_area_dy, c_lon, c_lat, effective_dx, effective_dy, elem_order, chunks, mesh, abg, uvrotated)

    ds = _TransportAcross(ds)

    if add_TS:
        ds = _AddTempSalt(section, ds, data_path, mesh, years, elem_order)

    ds = _OrderIndices(ds, elem_order)

    ds = _MaskBathymetry(ds, how)

    return  ds, section
