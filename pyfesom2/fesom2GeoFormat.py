from pyfesom2 import fesom2regular
from osgeo import osr,gdal
import numpy as np 








def fesom2GeoFormat(data,mesh,outName,radius_of_influence=100000,methode='nn', driver='Gtiff'):
    """ Returns Fesom as Gdal Geotiff or Gdal NetCDF .

    Parameters:
        data (array): which is come from select_slices().
        mesh (fesom mesh): which is come from load_mesh().
        outName (str): output Name.
        methode : interpolation methode for Fesom2Regular().
        driver(str): driver name for Gdal ( Gtiff or NetCDF).

    Returns:
        Gdal Datasets:Gtiff or NetCDF.   

    """
    lonreg,latreg,rasterOrigin = grid(mesh)
    data2= dataShape(data)
    pixelWidth = 1
    pixelHeight = -1
    outputName = outName
    array =  fesom2regular(
            data2,
            mesh,
            lonreg,
            latreg,
            distances_path=None,
            inds_path=None,
            qhull_path=None,
            how=methode,
            k=5,
            radius_of_influence=radius_of_influence,
            n_jobs=2,
            dumpfile=True,
            basepath=None,
            )
    
    reversed_arr = array[::-1] # reverse array so the tif looks like the array

    array2raster(
        outputName,
        rasterOrigin,
        pixelWidth,
        pixelHeight,
        reversed_arr,
        driver) #convert array to raster






def dataShape(data):
    if (len(data.shape)>1):
        data=data.flatten()
    else:
        data=data
    
    return data

def grid(mesh):
    xmin,ymin,xmax,ymax = [mesh.x2.min(),mesh.y2.min(),mesh.x2.max(),mesh.y2.max()]
    res=[int(round(abs(xmin)+abs(xmax))),int(round(abs(ymin)+abs(ymax)))]
    lonNumber, latNumber = res
    lonreg = np.linspace(xmin,xmax, lonNumber)
    latreg = np.linspace(ymin,ymax, latNumber)
    lonreg,latreg = np.meshgrid(lonreg,latreg)
    rasterOrigin = (xmin,ymax)
    return lonreg,latreg,rasterOrigin


def array2raster(outputName,rasterOrigin,pixelWidth,pixelHeight,array,driver='NetCDF'):

    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName(driver)
    outRaster = driver.Create(outputName, cols, rows, 1, gdal.GDT_Byte)
#   outRaster.SetGeoTransform(geotransform)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()