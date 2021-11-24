import os
import gdal
import ogr
import numpy
import random

from pathlib import Path
from sklearn.neural_network import MLPRegressor
from joblib import dump, load


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


###################################################################################################
def getTrainingData(workDir, trainingShp, bands):
    '''
    Obtiene los datos para entrenar una red neuronal a partir de un conjunto de datos raster con 
    varias bandas y de una capa vectorial de poligonos dada en un archivo SHP que describe las
    zonas de entrenamiento

    Args:
        workDir (string): directorio donde están los datos, tanto raster como la capa vectorial
        trainingShp (string): nombre de la capa vectorial
        bands (list): lista de nombres de los archivos del conjunto de datos raster

    Returns:
        (list): lista de vectores de entradas de entrenamiento
        (list): lista de vectores de salidas de entrenamiento
        (int): número de clases que describe la capa vectorial
    '''
    trainingShp =os.path.join(workDir, trainingShp)                 # completar la ruta del archivo shp de entrenamiento

    tmpDir = os.path.join(workDir, 'mlpc_tmp')                      # crear el directorio temporal para trabajar
    Path(tmpDir).mkdir(parents=True, exist_ok=True)

    # verificar que la capa de entrenamiento tenga el campo de clase
    print(trainingShp)
    trainingDS = ogr.Open(trainingShp, 0)                           # abrir el archivo shp de entrenamiento
    trainingLyr = trainingDS.GetLayer(0)                            # obtener la capa del archivo
    lyrDef = trainingLyr.GetLayerDefn()                             # obtener los metadatos de la capa

    fields = [lyrDef.GetFieldDefn(i).GetName() for i in range(lyrDef.GetFieldCount())]  # obtener los nombres de los campos de la tabla de atributos
    classFldIdx = -1                                                # índice donde se encuentra el nombre del campo de clase
    for idx, name in enumerate(fields):                             # buscar el campo de clase
        if name == 'clase':
            classFldIdx = idx

    if classFldIdx < 0:                                             # si no se encontró el campo de clase
        print('No class attribute')                                 # reportarlo y salir
        return -1                                             

    # calcular el número de clases que tiene la capa de entrenamiento
    classes = 0                                                     # número de clases
    for feat in trainingLyr:                                        # para cada poligono en la capa de entrenamiento
        featClass = feat.GetField('clase')                          # obtener el campo de clase
        if featClass > classes:                                     # y calcular el máximo valor de clase
            classes = featClass

    print('Clases:', classes)

    # generar los clips con las bandas
    for clase in range(1, classes+1):                               # para cada clase posible
        print('Processing class', clase, ' ...', end='') 
        for band in bands:                                          # para cada banda en el conjuntto de datos raster
            print('band', band, end=', ')
            in_raster = os.path.join(workDir, band + '.tif')        # crear el nombre del raster de entrada
            out_raster = os.path.join(tmpDir, band + '_class_'+ str(clase) + '.tif')    # y el nombre del raster de salida

            # Ya se ha creado el clip?
            if not os.path.isfile(out_raster):                      # el archivo ya existe?
                options = gdal.WarpOptions(cutlineDSName=trainingShp, cutlineWhere='clase='+str(clase)) # no, crearlo
                gdal.Warp(out_raster, in_raster, options=options)
            else:
                print('also exists.', end=' ')                      # si, reportarlo
        print('Ok.')

    # extraer los datos de entrenamiento
    trainInputs = []                                                # arreglo con las entradas de entrenamiento
    trainOutputs = []                                               # arreglo con las salidas de entrenamiento
    for clase in range(1, classes + 1):                             # para cada clase
        trainRasterLayers = []                                      # generar una lista con los raster recortados de esa clase
        for band in bands:                                          # para cada banda
            trainRasterLayers.append(os.path.join(tmpDir, band + '_class_'+ str(clase) + '.tif')) # agregar el raster a la lista

        ti, to = extractTrainData(trainRasterLayers, clase, classes)# extraer los datos de entradas y salidas de entrenamiento
        trainInputs = trainInputs + ti                              # agregar al arreglo de entradas de entrenamiento
        trainOutputs = trainOutputs + to                            # agregar al arreglo de salidas de entrenamiento

    # agregar datos de entrenamiento para la clase 0
    nZeros = round(len(trainInputs) * 0.05)
    print('Generating ', nZeros, ' zero vectors.', end='')
    inZeros = [[random.random() / 100000 for _ in range(len(bands))] for _ in range(nZeros)]   
    outZeros = [0] * nZeros
    # inZeros = [[0 for _ in range(len(bands))] for _ in range(nZeros)]   
    # outZeros = [0 for _ in range(nZeros)]
    trainInputs = trainInputs + inZeros
    trainOutputs = trainOutputs + outZeros
    print('Ok.')

    # desordenar los datos de entrenamiento
    train = list(zip(trainInputs, trainOutputs))                    # unimos ambas listas para vincular entrads con salidas
    random.shuffle(train)                                           # desordenamos la lista unida
    trainInputs, trainOutputs = zip(*train)                         # y separamos las entradas de las salidas

    return trainInputs, trainOutputs, classes


###################################################################################################
def mlpTest(testInputs, testOutputs, model):
    '''
    Ejecuta una prueba al modelo de la red neuronal

    Args:
        testInputs (list): lista de vectores de datos de entradas de prueba
        testOutputs (list): lista de vectores de datos de salidas de prueba
        model (): modelo de la red neuronal entrenada

    Returns:
        (list): lista de salidas predichas por el modelo
        (float): error cuadrático medio de la prueba
    '''
    print('Predicting ... ', end='')
    predictedOutputs = model.predict(testInputs)                    # clasificamos con los datos de prueba
    print('Ok')
    
    mse = 0                                                         # error cuadrático medio
    for i in range(len(testOutputs)):                               # para cada valor de los conjuntos de salida
        mse += (testOutputs[i] - predictedOutputs[i]) ** 2          # calcular el error cuadrático
    mse /= len(testOutputs)                                         # obtener el promedio
    print('MSE:', mse)

    return predictedOutputs, mse


###################################################################################################
def mlpTraining(trainInputs, trainOutputs, 
        hidden_layer_sizes=[2,16,8,1], 
        solver='sgd', 
        max_iter=100, 
        learning_rate=0.002, 
        verbose=False,
        n_iter_no_change=10,
        batch_size=64
    ):
    # print('Entrenando ... ', end='', flush=True)
    print('Training ... ', end='')
    model = MLPRegressor(                                           # crear el modelo de perceptrón multicapa
        solver=solver, 
        max_iter=max_iter,
        learning_rate_init=learning_rate, 
        hidden_layer_sizes=tuple(hidden_layer_sizes[1:]),           # quitamos el primer valor, ya que este está dado por el tamaño del vector de entrada
        verbose=verbose, 
        n_iter_no_change=n_iter_no_change, 
        batch_size = batch_size
    )

    model.fit(trainInputs, trainOutputs)                            # ejecutar el ciclo de entrenamiento
    print('Ok')

    return model                                                    # regresar el modelo


###################################################################################################
def extractTrainData(rasterLayers, clase, classes, stride=10):
    bands = []                                                      # las bandas como arreglos
    maxBandVal = []                                                 # valores máximos de cada banda
    nBands = len(rasterLayers)                                      # número de bandas
    for rasterLayer in rasterLayers:                                # para cada capa raster
        rasterData = gdal.Open(rasterLayer)                         # abrir el raster
        rasterArray = rasterData.GetRasterBand(1).ReadAsArray()     # leer los datos como arreglo
        bands.append(rasterArray)                                   # agregar los datos leidos al arreglo de bandas
        maxBandVal.append(numpy.amax(rasterArray))                  # calcular y agregar el máximo de la banda

    height, width = bands[0].shape                                  # obtenemos el ancho y alto
    trainInputs = []                                                # arreglo de entradas de entrenamiento
    trainOutputs = []                                               # arreglo de salidas de entrenamiento

    for y in range(0, height, stride):                              # recorrer los renglones con saltos de tamaño stride
        if y % round(height / 10) < stride:
            print('\rExtracting data of class', clase, end=' ')
            print( round(100 * y / height),'% ', sep='', end='')
            
        for x in range(0, width, stride):                           # recorrer los pixeles del renglón con saltos de tamaño stride
            pxVal = 0                                               # suma de los valores del pixel en cada banda
            # pxVec = [x / width, y / height]                         # incluir los índices en el entrenamiento
            pxVec = []                                              # vector de entrenamiento
            for band in range(nBands):                              # para cada banda
                pxVal += bands[band][y, x]                          # sumamos el valor del pixel en esa banda
                pxVec.append(bands[band][y, x] / maxBandVal[band])  # agregamos el valor del pixel en esa banda al vector de entrenamiento
                
            if pxVal > 0:                                           # si el pixel tiene valor
                trainInputs.append(pxVec)                           # lo agregamos a las entradas de entrenamiento
                trainOutputs.append(clase / classes)                # y la salida corresponde a la clase normalizada

    print('\rExtracting data of class', clase, 'Ok.', len(trainInputs))

    return trainInputs, trainOutputs                                # regresamos las entradas y salidas de entrenamiento

###################################################################################################
def saveToCSV(filename, data):
    import csv

    with open(filename, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(data)


###################################################################################################
def mlpClassification(workDir, rasterLayers, model, classes, resFilename='result.tif'):
    import time

    start = time.time()
    print('Starting classification ...')
    bands = []                                                      # lista con las bandas para la clasificación
    width = 1                                                       # tamaño en x
    height = 1                                                      # tamaño en y

    for rasterLayer in rasterLayers:                                # para cada capa raster en la clasificacion
        print('Processing raster', rasterLayer, '...', end='')
        rasterData = gdal.Open(os.path.join(workDir, rasterLayer + '.tif')) # abrir el archivo original
        rasterArray = rasterData.GetRasterBand(1).ReadAsArray()     # leemos los datos como un arreglo
        transform = rasterData.GetGeoTransform()                    # obtener la transformación geográfica
        projection = rasterData.GetProjection()                     # obtener la proyección geográfica

        maxBandVal = numpy.amax(rasterArray)                        # calculamos el valor máximo de la banda para normalizar
        height, width = rasterArray.shape                           # guardamos el tamaño original del arreglo
        rasterArray = rasterArray.reshape(-1)                       # convertimos en un arreglo de 1 dimensión
        bands.append([(val / maxBandVal) for val in rasterArray])   # agregamos la el arreglo normalizado a la lista de bandas
        print('Ok.')

    # print('Calculando Xs y Ys ...')
    # ys = numpy.array([[y / height] * width for y in range(height)]) # para los valores de las coordenadas verticales
    # xs = numpy.transpose(numpy.array([[x / width] * height for x in range(width)])) # para los valores de las coordenadas horizontales
    # ys = ys.reshape(-1)                                             # convertirlos a un arreglo unidimensional
    # xs = xs.reshape(-1)

    # bands.insert(0, ys)                                             # agregarlos al inicio de la lista de bandas para preservar
    # bands.insert(0, xs)                                             # el orden para la clasificación (x, y, b0, b1, b2, ...)

    batchCount = height                                             # número de lotes
    batchSize = width                                               # calcular el tamaño de cada lote
    outBand = []                                                    # arreglo de salida
    for b in range(batchCount):                                     # para cada lote 
        if b % round(batchCount / 100) == 0:
            print('\rClassifying ...',round(100 * b / batchCount, 2), '%', end='')    
        batch = list(zip(*[band[(b * batchSize):((b + 1) * batchSize)] for band in bands]))   # crear el lote   
        outBand.append(model.predict(batch)) 
    print('\rClassifying ... Ok.            ')

    # print('Postprocessing ...', end='', flush=True)
    print('Postprocessing ...', end='')
    outBand = numpy.array(outBand)                                  # convertir a numpy array
    outBand = outBand / numpy.amax(outBand)
    outBand = numpy.array([[round(classes * x) for x in row] for row in outBand])    # establilizar en las clases
    print('Ok.')

    # Escribir raster
    writeArrayToTIFF(os.path.join(workDir,resFilename), outBand, transform, projection)

    end = time.time()
    print('Process ended in', round(end - start, 1), 's.')


###################################################################################################
def main():
    workDir = '/home/rolando/Descargas/raster/MLPClassifier'
    trainingShp = 'train1.shp'
    name = '48881'
    modelFile = name + '_modelo.mlp'
    tifFile = name + '_resultado'
    bands = ['B1', 'B2', 'B3', 'B4']
    clases = 4
    testSize = 0.2
    nnShape = [4, 8, 8, 8, 1]

    print('Loading model ... ', end='')
    if Path(os.path.join(workDir, modelFile)).is_file():
        modelo = load(os.path.join(workDir, modelFile))
        clases = 4
        print('Ok.')
    else:
        ti, to, clases = getTrainingData(workDir, trainingShp, bands)

        # separar conjunto de entrenamiento y de prueba
        sep = round(len(ti) * (1 - testSize))                           # calculamos el punto de separación
        testInputs = ti[sep:]                                           # separamos las entradas de prueba
        testOutputs = to[sep:]                                          # separamos las salidas de prueba
        trainInputs = ti[0:sep]                                         # nos quedamos con las entradas de entrenamiento
        trainOutputs = to[0:sep]                                        # nos quedamos con las salidas de entrenamiento
        
        modelo = mlpTraining(trainInputs, trainOutputs, hidden_layer_sizes=nnShape) # ejecutar el entrenamiento

        po, mse = mlpTest(testInputs, testOutputs, modelo)              # ejecutar la prueba

        print('Saving model ... ', end='')
        dump(modelo, os.path.join(workDir, modelFile))                  # volcar el modelo
        print('Ok.')

        print('Saving CSV ... ', end='')
        outs = zip(testInputs, testOutputs, po)                                     # unir ambos conjuntos de salida
        saveToCSV(os.path.join(workDir, 'outs.csv'), outs)              # guardarlos en un CSV
        print('Ok.')

    zero = modelo.predict([[0,0,0,0]])
    print(zero)
    if zero < 0.1:
        mlpClassification(workDir, bands, modelo, clases, resFilename=tifFile)


###################################################################################################
# main()


