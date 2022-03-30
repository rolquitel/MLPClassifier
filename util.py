from qgis.PyQt.QtCore import *
from qgis.PyQt.QtGui import *
from qgis.PyQt.QtWidgets import *

import gdal

###################################################################################################
def writeArrayToTIFF(filename, outArray, transform, projection, noData=0):
    '''
    Guarda un arreglo raster en un archivo TIF

    Args:
        filename (string): nombre del archivo para guardar
        outArray (numpy.array): arreglo con los datos del raster
        transform (GeoTransform): geo transformación del raster
        projection (GeoProjection): geo proyección del raster
        noData (int, optional): valor para el valor que representa el dato nulo. Defaults to 0.
    '''       
    print('Guardando resultado ...', end='')
    sx, sy = outArray.shape
    driver = gdal.GetDriverByName("GTiff")
    tiff = driver.Create(filename, sy, sx, 1, gdal.GDT_UInt16)
    tiff.SetGeoTransform(transform)       
    tiff.SetProjection(projection)           
    tiff.GetRasterBand(1).WriteArray(outArray)
    tiff.GetRasterBand(1).SetNoDataValue(noData)      
    tiff.FlushCache()    
    print('Ok.')

class Logger:
    def __init__(self, edit, out=None, color=None):
        """(edit, out=None, color=None) -> can write stdout, stderr to a
        QTextEdit.
        edit = QTextEdit
        out = alternate stream ( can be the original sys.stdout )
        color = alternate color (i.e. color stderr a different color)
        """
        self.edit = edit
        self.out = None
        self.color = color

    def write(self, m):
        if self.color:
            tc = self.edit.textColor()
            self.edit.setTextColor(self.color)

        self.edit.moveCursor(QTextCursor.End)
        self.edit.insertPlainText( m )

        if self.color:
            self.edit.setTextColor(tc)

        if self.out:
            self.out.write(m)

        self.edit.repaint()             # importante para no tener que esperar el ciclo de refresco de QT

def get_shape(shape):
    try:
        layers = shape.split(' ')
        retVal = [0]
        for layer in layers:
            retVal.append(int(layer))

        retVal.append(1)
        return retVal
    except:
        return None
