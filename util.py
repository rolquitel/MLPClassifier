import os
import gdal
import numpy

def rasterIO(filename):
    ds = gdal.Open(filename)                            # Abrir raster
    band = ds.GetRasterBand(1)                          # Leer la primera banda
    arr = band.ReadAsArray()                            # Leer la banda como arreglo
    [rows, cols] = arr.shape                            # Calcular número filas y columnas

    # Procesamiento del raster
    arr_min = arr.min()                                 # Calcular mín y max del arreglo
    arr_max = arr.max()
    arr_mean = int(arr.mean())                          # Calcula el promedio
    arr_out = numpy.where((arr < arr_mean), 10000, arr)

    # Escribir raster
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(filename + "_out", cols, rows, 1, gdal.GDT_UInt16)
    outdata.SetGeoTransform(ds.GetGeoTransform())       ##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())           ##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(arr_out)
    outdata.GetRasterBand(1).SetNoDataValue(10000)      ##if you want these values transparent
    outdata.FlushCache()                                ##saves to disk!!
    outdata = None
    band=None
    ds=None

def writeArrayToTIFF(filename, outArray, transform, projection, noData=0, verbose=False):
    echo('Guardando resultado ...', end='', verbose=verbose)
    sx, sy = outArray.shape
    driver = gdal.GetDriverByName("GTiff")
    tiff = driver.Create(filename, sy, sx, 1, gdal.GDT_UInt16)
    tiff.SetGeoTransform(transform)       
    tiff.SetProjection(projection)           
    tiff.GetRasterBand(1).WriteArray(outArray)
    tiff.GetRasterBand(1).SetNoDataValue(noData)      
    tiff.FlushCache()    
    echo('Ok.', verbose=verbose)

def echo(message, verbose=False, end='\n'):
    if verbose:
        print(message, end=end)