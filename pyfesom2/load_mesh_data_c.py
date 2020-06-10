# -*- coding: utf-8 -*-
"""
Created on Mon Apr  27 12:00:00 2020

@author: Ivan Kuznetsov 
"""
#
# This file is part of pyfesom2: https://github.com/FESOM/pyfesom2.git
# Original code by Ivan Kuznetsov, 2020, folowing FESOM2 code structure from Dmitry Sidorenko, 2013
#


import pandas as pd
import numpy as np
from netCDF4 import Dataset
import os
import logging
from netCDF4 import num2date
#from cftime import  num2pydate
import datetime
import matplotlib.pyplot as plt
import joblib
import pickle
import pyresample
#import nc_time_axis
from   matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def load_c_mesh(path, exp = "", usepickle=False, usejoblib=False, protocol=4, addpolygons=False):
    """ Loads FESOM-C mesh
    
    fesom_c v.2 save mesh in two ways. first is *.out ascii files, that model is reading.
    second, fesom_c write mesh to output netcdf files (each full 3d output, has mesh inside netcdf).
    fesom_c v.2x,3, mpi - will write it diferntly , maybe one netcdf file for mesh.

    Parameters
    ----------
    path : str
        Path to the directory with mesh files
    usepickle (optional): bool
        use pickle file to store or load mesh data
    usejoblib (optional): bool
        use joblib file to store or load mesh data
    protocol (optional): int
        used for pickle, only way to save data more than 4 Gb
    Returns
    -------
    mesh : object
        fesom_c_mesh object
    """
    path = os.path.abspath(path)
    if (usepickle == True) and (usejoblib == True):
        raise ValueError(
            "Both `usepickle` and `usejoblib` set to True, select only one"
        )

    if usepickle:
        pickle_file = os.path.join(path, "pickle_mesh_py3_fesom_c")
        print(pickle_file)

    if usejoblib:
        joblib_file = os.path.join(path, "joblib_mesh_fesom_c")

    if usepickle and (os.path.isfile(pickle_file)):
        print("The usepickle == True)")
        print("The pickle file for FESOM_C exists.")
        print("The mesh will be loaded from {}".format(pickle_file))

        ifile = open(pickle_file, "rb")
        mesh = pickle.load(ifile)
        ifile.close()
        return mesh

    elif (usepickle == True) and (os.path.isfile(pickle_file) == False):
        print("The usepickle == True")
        print("The pickle file for FESOM_C DO NOT exists")
        print("The mesh will be saved to {}".format(pickle_file))

        mesh = fesom_c_mesh(path=path, exp=exp)
        logging.info("Use pickle to save the mesh information")
        print("Save mesh to binary format")
        outfile = open(pickle_file, "wb")
        pickle.dump(mesh, outfile, protocol=protocol)
        outfile.close()
        return mesh

    elif (usepickle == False) and (usejoblib == False):
        mesh = fesom_c_mesh(path=path, exp=exp)
        return mesh

    if (usejoblib == True) and (os.path.isfile(joblib_file)):
        print("The usejoblib == True)")
        print("The joblib file for FESOM_C exists.")
        print("The mesh will be loaded from {}".format(joblib_file))

        mesh = joblib.load(joblib_file)
        return mesh

    elif (usejoblib == True) and (os.path.isfile(joblib_file) == False):
        print("The usejoblib == True")
        print("The joblib file for FESOM_C DO NOT exists")
        print("The mesh will be saved to {}".format(joblib_file))

        mesh = fesom_c_mesh(path=path, exp=exp, addpolygons=False)
        logging.info("Use joblib to save the mesh information")
        print("Save mesh to binary format")
        joblib.dump(mesh, joblib_file)

        return mesh

class fesom_c_mesh(object):
    """ Creates instance of the FESOM-C mesh.
    This class creates instance that contain information
    about FESOM-C mesh. At present the class works with
    ASCII representation of the FESOM-C grid, 
    it read also netCDF version of FESOM-C.
    while reading ASCII version no information about sigma is loaded.
    NetCDF is preferably. 

    Minimum requirement is to provide the path to the directory and 
    <expname> experiment name,
    where following files should be located :

    - <expname>_nod2d.out
    - <expname>_elem2d.out
    - <expname>_depth.out

    Parameters
    ----------
    path : str
        Path to the directory with mesh files 
           OR
        if path ends on ".nc", netcdf file will be used to read mesh

    exp : str (optional for netcdf file)
        name of the experiment (<exp>_nod2d.out)
        if no exp is provided for ascii case or exp = "" than files:
            nod2d.out, ... will be used
        

    Attributes
    ----------
    path : str
        Path to the directory with mesh files
    x2 : array
        x position (lon) of the surface node
    y2 : array
        y position (lat) of the surface node
    n2d : int
        number of 2d nodes
    e2d : int
        number of 2d elements (triangles)
    type : str
        type of mesh (fesom-c for FESOM-C)
        
    Returns
    -------
    mesh : object
        fesom_mesh object
    """

    def __init__(self, path, exp="", addpolygons=False, readTime=True):
        #add type of mesh
        #addpolygons - will construct polygons based on mesh xy
        #readTime - read time from nc file, it is no related to mesh but still usefull
        #           later on could be moved to data file class
        self.type = 'fesom_c'

        #find if nectdf file is provided 
        if (path[-3:] != '.nc'):
            useascii = True
            usenetcdf = False
            self.path = os.path.abspath(path)
        else:
            useascii = False
            usenetcdf = True
            s = os.path.abspath(path)
            self.path = os.path.dirname(s)
        
        if not os.path.exists(path):
            raise IOError('The path/file "{}" does not exists'.format(path))
        #predifinition. (why?)
        self.e2d = 0
        self.nlev = 0
        self.zlevs = []
        self.topo = []

        if useascii:    
            s = ""
            if (exp != ""):
                s=exp+"_"
            self.nod2dfile = os.path.join(self.path, s+"nod2d.out")
            self.elm2dfile = os.path.join(self.path, s+"elem2d.out")
            self.depth2dfile = os.path.join(self.path, s+"depth.out")
                
            if not os.path.exists(self.nod2dfile):
                raise IOError('The file "{}" does not exists'.format(self.nod2dfile))
            if not os.path.exists(self.elm2dfile):
                raise IOError('The file "{}" does not exists'.format(self.elm2dfile))
            if not os.path.exists(self.depth2dfile):
                raise IOError('The file "{}" does not exists'.format(self.depth2dfile))
        else:
            self.ncfile = os.path.join(path)            

        logging.info("load 2d part of the mesh")
        #start = time.clock()
        if useascii:
            self.read2d()
        else:
            self.read2d_nc(readTime=readTime)
        # add table of elements with coordinates    
        self.elem_x = self.x2[self.elem-1]
        self.elem_y = self.y2[self.elem-1] 
        # add polygons of mesh (could take time for huge meshes , and  MEMORY)
        if (addpolygons):
            self.addpolygons() 
            
        #end = time.clock()
        #print("Load 2d part of the mesh in {} second(s)".format(str(int(end - start))))
        
    def addpolygons(self):
        #function add polygons , used later for ploting
        p=[Polygon(np.vstack((self.elem_x[i], self.elem_y[i])).T,closed=True) 
                        for i in range(self.e2d)]
        self.patches = p
        return
       
    def read2d(self):
        # funcion to read mesh files for FESOM-C branch, 
        # * it has 4 nodes elements in each element
        # * it uses sigma vertical discretization
        # * no aux file, but depth
        file_content = pd.read_csv(
            self.nod2dfile,
            delim_whitespace=True,
            skiprows=1,
            names=["node_number", "x", "y", "flag"],
        )
        self.x2 = file_content.x.values
        self.y2 = file_content.y.values
        self.ind2d = file_content.flag.values
        self.n2d = len(self.x2)

        file_content = pd.read_csv(
            self.elm2dfile,
            delim_whitespace=True,
            skiprows=1,
            names=["first_elem", "second_elem", "third_elem", "fourth_elem"],
        )
        self.elem = file_content.values - 1
        self.e2d = np.shape(self.elem)[0]
        
        #read depths
        file_content = pd.read_csv(
            self.depth2dfile,
            delim_whitespace=True,
            skiprows=0,
            names=["depth"],
        )
        self.topo = file_content.depth.values
               

        ###########################################
        # computation of volumes skiped here (it is done in FESOM-C output nc files)
        # compute the 2D lump operator skiped for fesom_c for now

        return self

    def read2d_nc(self, readTime=True):
        # funcion to read mesh files for FESOM-C branch. NetCDF version 
        # * it has 4 nodes elements in each element
        # * it uses sigma vertical discretization
        # * no aux file, but depth
        # variables will be added to object (mesh) if availeble in nc file,
        # personaly I do not like defind so many variable like self.<name>
        # i would prefer self.aux = {}
        #    self.aux.update({"varname":value})
        #    so to call it: mesh.aux['area']
        #    how does it works with paralel libs ?
        #readTime - will add time series from nc file to mesh object
        def loadvar(var):
            if (var in ncf.variables):
                data = ncf.variables[var][:].data
            return data
            
        ncf = Dataset(self.ncfile)
        
        self.x2 = loadvar('lon')
        self.y2 = loadvar('lat')
        self.elem = loadvar('nv')
        self.topo =  loadvar('depth')
        self.x2_e = loadvar('lon_elem')
        self.y2_e = loadvar('lat_elem')
        self.topo_e =  loadvar('depth_elem')
        self.sigma_lev = loadvar('sigma_lev')
        self.topo_e = loadvar('depth_elem')
        self.area = loadvar('area')
        self.elem_area = loadvar('elem_area')
        self.w_cv = loadvar('w_cv')
        self.nod_in_elem2d_num = loadvar('nod_in_elem2d_num')
        self.nod_in_elem2d = loadvar('nod_in_elem2d')
            #time
        if readTime:
            if ('time' in ncf.variables):
                mtime_raw = ncf.variables['time'][:]
                a = ncf.variables['time'].getncattr('units')
                # with new netCdf num2date works diferently 
                self.mcftime = num2date(mtime_raw,a)   
                self.mtime = [datetime.datetime(year=b.year,month=b.month,day=b.day,
                                        hour=b.hour,minute=b.minute,second=b.second,
                                        microsecond=b.microsecond) for b in self.mcftime]
                self.mtimec = np.array([t.timestamp() for t in self.mtime])
        self.e2d = np.shape(self.elem)[0]
        self.n2d = len(self.x2)
        ncf.close()

        return self


        
def plotpatches(mesh, data, figsize=(10, 7), dpi=90, title="", var="",
                  vmin=None,vmax=None,cont=None, Nlev=21, edge="face",linewidth=0.8):
    # if no patches were done, do it first time and add to mesh
    if not hasattr(mesh, 'patches'):
        mesh.addpolygons()
    d = data.copy()
    if (vmin==None):
        vmin = np.nanmin(data)
    if (vmax==None):
        vmax = np.nanmax(data)
    d[d>=vmax] = vmax
    d[d<=vmin] = vmin
    cmap = select_cmap(var)    
    fig, axes = plt.subplots(nrows=1, ncols=1,figsize=figsize, dpi=dpi)
    p = PatchCollection(mesh.patches,linewidth=linewidth,cmap=cmap)    
    plot = axes.add_collection(p)
    plot.set_array(d)
    plot.set_edgecolor(edge)  
    plot.set_clim(vmin,vmax)
    axes.grid(color='k', alpha=0.5, linestyle='--')
    axes.set_xlabel(r'$Longitude, [\degree]$',fontsize=18)  
    axes.set_ylabel(r'$Latitude, [\degree]$',fontsize=18)  
    axes.tick_params(labelsize=18)
    cbar = fig.colorbar(plot, aspect=40,
                        ticks=np.linspace(vmin,vmax,7))  
    cbar.ax.tick_params(labelsize=18)
    axes.set_title(title,fontsize=18)
    fig.autofmt_xdate()
    axes.autoscale_view()                
    return {'fig': fig, 'axes':axes, 'plot':plot}       
        
def read_fesomc_slice(
        fname,
        var,
        records,
        how="mean"
        ):
    # read data from FESOM-C 3d file
    # records - time steps
    ncf = Dataset(fname)
    if (type(records) == int):
        data = ncf.variables[var][records,:,:].data
    else:    
        if how == "mean":
            data = ncf.variables[var][records,:,:].data.mean(axis=0)
        elif how == "max":
            data = ncf.variables[var][records,:,:].data.max(axis=0)
        elif how == "min":
            data = ncf.variables[var][records,:,:].data.min(axis=0)
    ncf.close()
    return data

def sigma2z3d(data,mesh,z):
    #function for interpolation of 3d data on sigma level to 3d z levels
    #depth levels are calculated from bathymerty (topo) and sigma distribution
    #if you nead real depth (depth+ ssh) have a look on zbar variable in nc output
    
    if (mesh.x2.shape[0] == data.shape[0]):
        # if data on nodes
        topo = mesh.topo
    elif (mesh.x2_e.shape[0] == data.shape[0]):
        # if data on elements
        topo = mesh.topo_e
    else:
        raise IOError('Shape of data "{}" does not fit to nodes or elements'.format(data.shape))
    # matrix of depth levels    
    s,d = np.meshgrid((1-mesh.sigma_lev),topo)
    z0 = d*s #2d array with z levels at each node
    if data.shape[1] <len(mesh.sigma_lev):
        #if data on Nsigma-1 levels
        z0 = (z0[:,0:-1]+z0[:,1:])/2.0
    data_intp = np.zeros((data.shape[0],len(z)))
    data_intp[:,:] = np.nan
    #z_intp = np.zeros(data.shape[0])
    #loop over nodes/elements, interpolate from sigma to z,
    #excluding extrapolations
    # np.interp works diferently in python3.6 and 3.8 ???
    for i in range(data.shape[0]):
        zind = np.where(z < topo[i])[0][-1]+2
        data_intp[i,:zind] = np.interp(z[:zind],z0[i,:],data[i,:])
        #z_intp[i] = zind
    return data_intp    

def read_fesomc_sect(
        fname,
        mesh,
        var,
        records,
        p1,
        p2,
        how="mean",        
        Nz=30,
        N=100,
        radius_of_influence=5000,
        neighbours=10):        
    #function similar to fesom2, but taking into account sigma coordinates
    # set the number of descrete points in horizontal and vertical (N and nz, respectively) to represent the section
    # parameters:
    #   p1,p2 - start and end of section
    #   fname - name of file with data
    #   var - name of variable
    #   records - time steps
    #   how - if more than 1 record, mean,min or max over time steps 
    #   Nz - number of z levels
    #   N - number of horisontal steps
    #   radius_of_influence,neighbours
    #return:
    # sx,- coordinates of x
    # sy - y coordinates of section
    # sz - data resampled on section
    # z  - z levels of section
    # TODO:
    #   now new z is a linspace between 1 and maximum depth of mesh,
    #     it should be defined as parameter
    
    # sx,sy coordinates of section
    sx = np.linspace(p1[0], p2[0], N)
    sy = np.linspace(p1[1], p2[1], N)
    sz = np.zeros([N, Nz])
    sz[:,:] = np.nan
    # construct new z levels
    z = np.linspace(1,mesh.topo.max(),Nz)
    # read data from file
    data =  read_fesomc_slice(fname, var, records, how=how)
    # interpolate data from sigma to z levels
    data_intp = sigma2z3d(data,mesh,z)    
    # if on variable on nodes or elements
    if (mesh.x2.shape[0] == data.shape[0]):
        lons = mesh.x2
        lats = mesh.y2
    elif (mesh.x2_e.shape[0] == data.shape[0]):
        lons = mesh.x2_e
        lats = mesh.y2_e
    else:
        raise IOError('Shape of data "{}" does not fit to nodes or elements'.format(data.shape))
    # magic of pyresample start        
    oce_ind2d = np.ones(lons.shape)
    orig_def = pyresample.geometry.SwathDefinition(lons=lons, lats=lats)
    targ_def = pyresample.geometry.SwathDefinition(lons=sx, lats=sy)
    oce_mask = pyresample.kd_tree.resample_nearest(
            orig_def,
            oce_ind2d,
            targ_def,
            radius_of_influence=radius_of_influence,
            fill_value=0.0,
            )
    # do resampling level by level
    for ilev in range(Nz):
        sz[:,ilev] = (
                pyresample.kd_tree.resample_gauss(
                        orig_def,
                        data_intp[:,ilev],
                        targ_def,
                        radius_of_influence=radius_of_influence,
                        neighbours=neighbours,
                        sigmas=250000,
                        fill_value=np.nan,
                        )
                * oce_mask
                )
    return (sx, sy, sz, z)
                
