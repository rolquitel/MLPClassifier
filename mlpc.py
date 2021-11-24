import os
import gdal
import ogr
import numpy
import random

from pathlib import Path
from sklearn.neural_network import MLPRegressor
from joblib import dump, load

# from . import util
import util

###############################################################################
## Sampler
###############################################################################
class Sampler:
    '''
    Class to sample from a multiband image
    '''
    working_dir = ""
    training_shp = ""
    bands = []
    classes = 0
    train_inputs = []
    train_outputs = []
    test_inputs = []
    test_outputs = []
    stride = 10

    ###############################################################################
    def __init__(self, working_dir, training_shp, bands, stride=10):
        self.working_dir = working_dir
        self.training_shp = training_shp
        self.bands = bands
        self.stride = stride

    ###############################################################################
    def getTrainingData(self):
        '''
        Obtiene los datos para entrenar una red neuronal a partir de un conjunto de datos raster con 
        varias bandas y de una capa vectorial de poligonos dada en un archivo SHP que describe las
        zonas de entrenamiento
        '''
        self.training_shp =os.path.join(self.working_dir, self.training_shp)    # completar la ruta del archivo shp de entrenamiento

        tmpDir = os.path.join(self.working_dir, 'mlpc_tmp')                     # crear el directorio temporal para trabajar
        Path(tmpDir).mkdir(parents=True, exist_ok=True)

        # verificar que la capa de entrenamiento tenga el campo de clase
        print(self.training_shp)
        trainingDS = ogr.Open(self.training_shp, 0)                             # abrir el archivo shp de entrenamiento
        trainingLyr = trainingDS.GetLayer(0)                                    # obtener la capa del archivo
        lyrDef = trainingLyr.GetLayerDefn()                                     # obtener los metadatos de la capa

        fields = [lyrDef.GetFieldDefn(i).GetName() for i in range(lyrDef.GetFieldCount())]  # obtener los nombres de los campos de la tabla de atributos
        classFldIdx = -1                                                        # índice donde se encuentra el nombre del campo de clase
        for idx, name in enumerate(fields):                                     # buscar el campo de clase
            if name == 'clase':
                classFldIdx = idx

        if classFldIdx < 0:                                                     # si no se encontró el campo de clase
            print('No class attribute')                                         # reportarlo y salir
            return False                                          

        # calcular el número de clases que tiene la capa de entrenamiento
        self.classes = 0                                                        # número de clases
        for feat in trainingLyr:                                                # para cada poligono en la capa de entrenamiento
            featClass = feat.GetField('clase')                                  # obtener el campo de clase
            if featClass > self.classes:                                             # y calcular el máximo valor de clase
                self.classes = featClass

        print('Clases:', self.classes)

        # generar los clips con las bandas
        for clase in range(1, self.classes+1):                                  # para cada clase posible
            print('Processing class', clase, ' ...', end='') 
            for band in self.bands:                                             # para cada banda en el conjuntto de datos raster
                print('band', band, end=', ')
                in_raster = os.path.join(self.working_dir, band + '.tif')       # crear el nombre del raster de entrada
                out_raster = os.path.join(tmpDir, band + '_class_'+ str(clase) + '.tif')    # y el nombre del raster de salida

                # Ya se ha creado el clip?
                if not os.path.isfile(out_raster):                              # el archivo ya existe?
                    options = gdal.WarpOptions(                                 # no, crearlo
                        cutlineDSName=self.training_shp, 
                        cutlineWhere='clase='+str(clase))                       
                    gdal.Warp(out_raster, in_raster, options=options)
                else:
                    print('also exists.', end=' ')                              # si, reportarlo
            print('Ok.')

        # extraer los datos de entrenamiento
        for clase in range(1, self.classes + 1):                                # para cada clase
            trainRasterLayers = []                                              # generar una lista con los raster recortados de esa clase
            for band in self.bands:                                             # para cada banda
                trainRasterLayers.append(os.path.join(tmpDir, band + '_class_'+ str(clase) + '.tif')) # agregar el raster a la lista

            ti, to = self.extractTrainData(trainRasterLayers, clase)            # extraer los datos de entradas y salidas de entrenamiento
            self.train_inputs = self.train_inputs + ti                          # agregar al arreglo de entradas de entrenamiento
            self.train_outputs = self.train_outputs + to                        # agregar al arreglo de salidas de entrenamiento

        # agregar datos de entrenamiento para la clase 0
        nZeros = round(len(self.train_inputs) * 0.05)
        print('Generating ', nZeros, ' zero vectors.', end='')
        inZeros = [[random.random() / 100000 for _ in range(len(self.bands))] for _ in range(nZeros)]   
        outZeros = [0] * nZeros
        # inZeros = [[0 for _ in range(len(bands))] for _ in range(nZeros)]   
        # outZeros = [0 for _ in range(nZeros)]
        self.train_inputs = self.train_inputs + inZeros
        self.train_outputs = self.train_outputs + outZeros
        print('Ok.')

        self.shuffleData()

        return True

    ###############################################################################
    def shuffleData(self):
        '''
        Función para desordenar los datos de entrenamiento.
        '''
        train = list(zip(self.train_inputs, self.train_outputs))                # unimos ambas listas para vincular entrads con salidas
        random.shuffle(train)                                                   # desordenamos la lista unida
        self.train_inputs, self.train_outputs = zip(*train)                     # y separamos las entradas de las salidas

    ###############################################################################
    def extractTrainData(self, rasterLayers, clase):
        bands = []                                                              # las bandas como arreglos
        maxBandVal = []                                                         # valores máximos de cada banda
        nBands = len(rasterLayers)                                              # número de bandas

        for rasterLayer in rasterLayers:                                        # para cada capa raster
            rasterData = gdal.Open(rasterLayer)                                 # abrir el raster
            rasterArray = rasterData.GetRasterBand(1).ReadAsArray()             # leer los datos como arreglo
            bands.append(rasterArray)                                           # agregar los datos leidos al arreglo de bandas
            maxBandVal.append(numpy.amax(rasterArray))                          # calcular y agregar el máximo de la banda

        height, width = bands[0].shape                                          # obtenemos el ancho y alto
        trainInputs = []                                                        # arreglo de entradas de entrenamiento
        trainOutputs = []                                                       # arreglo de salidas de entrenamiento
        for y in range(0, height, self.stride):                                 # recorrer los renglones con saltos de tamaño stride
            if y % round(height / 10) < self.stride:
                print('\rExtracting data of class', clase, end=' ')
                print( round(100 * y / height),'% ', sep='', end='')
                
            for x in range(0, width, self.stride):                              # recorrer los pixeles del renglón con saltos de tamaño stride
                pxVal = 0                                                       # suma de los valores del pixel en cada banda
                # pxVec = [x / width, y / height]                                 # incluir los índices en el entrenamiento
                pxVec = []                                                      # vector de entrenamiento
                for band in range(nBands):                                      # para cada banda
                    pxVal += bands[band][y, x]                                  # sumamos el valor del pixel en esa banda
                    pxVec.append(bands[band][y, x] / maxBandVal[band])          # agregamos el valor del pixel en esa banda al vector de entrenamiento
                    
                if pxVal > 0:                                                   # si el pixel tiene valor
                    trainInputs.append(pxVec)                                   # lo agregamos a las entradas de entrenamiento
                    trainOutputs.append(clase / self.classes)                   # y la salida corresponde a la clase normalizada

        print('\rExtracting data of class', clase, 'Ok.', len(trainInputs))

        return trainInputs, trainOutputs    

    ###############################################################################
    def splitData(self, test_size):
        '''
        Función para dividir los datos de entrenamiento en entradas y salidas.
        '''
        sep = round(len(self.train_inputs) * (1 - test_size))
        self.test_inputs = self.train_inputs[sep:]
        self.test_outputs = self.train_outputs[sep:]

        self.train_inputs = self.train_inputs[0:sep]
        self.train_outputs = self.train_outputs[0:sep]

        print('Train data:', len(self.train_inputs), 'Test data:', len(self.test_inputs))


###############################################################################
## Classifier
###############################################################################
class Classifier:
    '''
    Clase para el clasificador.
    '''
    classes = 0                                                                 # número de clases

    def __init__(self, classes):
        '''
        Constructor.
        '''
        self.classes = classes

    ###############################################################################
    def test(self, testInputs, testOutputs):
        '''
        Ejecuta una prueba al modelo de la red neuronal

        Args:
            testInputs (list): lista de vectores de datos de entradas de prueba
            testOutputs (list): lista de vectores de datos de salidas de prueba

        Returns:
            (float): error cuadrático medio de la prueba
        '''
        print('Predicting ... ', end='')
        predictedOutputs = self.predict(testInputs)                             # clasificamos con los datos de prueba
        print('Ok')
        
        mse = 0                                                                 # error cuadrático medio
        for i in range(len(testOutputs)):                                       # para cada valor de los conjuntos de salida
            mse += (testOutputs[i] - predictedOutputs[i]) ** 2                  # calcular el error cuadrático
        mse /= len(testOutputs)                                                 # obtener el promedio
        print('MSE:', mse)

        return mse

    ###############################################################################
    def predict(self, inputs):
        '''
        Clasifica un conjunto de datos de prueba.

        Args:
            testInputs (list): lista de vectores de datos de entradas de prueba

        Returns:
            (list): lista de salidas predichas por el modelo
        '''
        return inputs

    ###############################################################################
    def training(trainInputs, trainOutputs):
        '''
        Entrena el modelo.

        Args:
            trainInputs (list): lista de vectores de datos de entradas de entrenamiento
            trainOutputs (list): lista de vectores de datos de salidas de entrenamiento
        '''
        pass

    ###############################################################################
    def classify(self, workDir, rasterLayers, resFilename='result.tif'):
        import time

        start = time.time()
        print('Starting classification ...')
        bands = []                                                              # lista con las bandas para la clasificación
        width = 1                                                               # tamaño en x
        height = 1                                                              # tamaño en y

        for rasterLayer in rasterLayers:                                        # para cada capa raster en la clasificacion
            print('Processing raster', rasterLayer, '...', end='')
            rasterData = gdal.Open(os.path.join(workDir, rasterLayer + '.tif')) # abrir el archivo original
            rasterArray = rasterData.GetRasterBand(1).ReadAsArray()             # leemos los datos como un arreglo
            transform = rasterData.GetGeoTransform()                            # obtener la transformación geográfica
            projection = rasterData.GetProjection()                             # obtener la proyección geográfica

            maxBandVal = numpy.amax(rasterArray)                                # calculamos el valor máximo de la banda para normalizar
            height, width = rasterArray.shape                                   # guardamos el tamaño original del arreglo
            rasterArray = rasterArray.reshape(-1)                               # convertimos en un arreglo de 1 dimensión
            bands.append([(val / maxBandVal) for val in rasterArray])           # agregamos la el arreglo normalizado a la lista de bandas
            print('Ok.')

        # print('Calculando Xs y Ys ...')
        # ys = numpy.array([[y / height] * width for y in range(height)]) # para los valores de las coordenadas verticales
        # xs = numpy.transpose(numpy.array([[x / width] * height for x in range(width)])) # para los valores de las coordenadas horizontales
        # ys = ys.reshape(-1)                                             # convertirlos a un arreglo unidimensional
        # xs = xs.reshape(-1)

        # bands.insert(0, ys)                                             # agregarlos al inicio de la lista de bandas para preservar
        # bands.insert(0, xs)                                             # el orden para la clasificación (x, y, b0, b1, b2, ...)

        batchCount = height                                                     # número de lotes
        batchSize = width                                                       # calcular el tamaño de cada lote
        outBand = []                                                            # arreglo de salida
        for b in range(batchCount):                                             # para cada lote 
            if b % round(batchCount / 100) == 0:
                print('\rClassifying ...',round(100 * b / batchCount, 2), '%', end='')    
            batch = list(zip(*[band[(b * batchSize):((b + 1) * batchSize)] for band in bands]))   # crear el lote   
            outBand.append(self.predict(batch)) 
        print('\rClassifying ... Ok.            ')

        # print('Postprocessing ...', end='', flush=True)
        print('Postprocessing ...', end='')
        outBand = numpy.array(outBand)                                          # convertir a numpy array
        outBand = outBand / numpy.amax(outBand)
        outBand = numpy.array([[round(self.classes * x) for x in row] for row in outBand])    # establilizar en las clases
        print('Ok.')

        # Escribir raster
        util.writeArrayToTIFF(os.path.join(workDir,resFilename), outBand, transform, projection)

        end = time.time()
        print('Process ended in', round(end - start, 1), 's.')

    ###############################################################################
    def save(self, filename):
        pass

    ###############################################################################
    def load(self, filename):
        pass


###############################################################################
## MLPR_Classifier
###############################################################################
class MLPR_Classifier(Classifier):
    model = None                                                         # modelo de perceptrón multicapa
    '''
    Clase para el clasificador MLP.
    '''
    def __init__(self, 
        classes, 
        hidden_layer_sizes=[2,16,8,1], 
        solver='sgd', 
        max_iter=100, 
        learning_rate=0.002, 
        verbose=False,
        n_iter_no_change=10,
        batch_size=64
    ):
        '''
        Constructor.
        '''
        super().__init__(classes)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.solver = solver
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.n_iter_no_change = n_iter_no_change
        self.batch_size = batch_size

    ###############################################################################
    def training(self, trainInputs, trainOutputs):
        '''
        Entrena el modelo de la red neuronal.

        Args:
            trainInputs (list): lista de vectores de datos de entradas de entrenamiento
            trainOutputs (list): lista de vectores de datos de salidas de entrenamiento

        Returns:
            (MLPRegressor): modelo de la red neuronal
        '''
        print('Training ... ', end='')
        self.model = MLPRegressor(                                              # crear el modelo de perceptrón multicapa
            solver = self.solver, 
            max_iter = self.max_iter,
            learning_rate_init = self.learning_rate, 
            hidden_layer_sizes = tuple(self.hidden_layer_sizes[1:]),            # quitamos el primer valor, ya que este está dado por el tamaño del vector de entrada
            verbose = self.verbose, 
            n_iter_no_change = self.n_iter_no_change, 
            batch_size = self.batch_size
        )

        self.model.fit(trainInputs, trainOutputs)                                    # ejecutar el ciclo de entrenamiento
        print('Ok')

    ###############################################################################
    def predict(self, inputs):
        '''
        Clasifica un conjunto de datos de prueba.

        Args:
            testInputs (list): lista de vectores de datos de entradas de prueba

        Returns:
            (list): lista de salidas predichas por el modelo
        '''
        return self.model.predict(inputs)

    ###############################################################################
    def save(self, filename):
        '''
        Guarda el modelo en un archivo.

        Args:
            filename (str): nombre del archivo
        '''
        dump(self.model, filename)
        # TODO: guardar los parámetros del modelo

    ###############################################################################
    def load(self, filename):
        '''
        Carga el modelo desde un archivo.

        Args:
            filename (str): nombre del archivo
        '''
        self.model = load(filename)
        # TODO: falta poder cargar el número de clases


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

    cl = None

    print('Loading model ... ', end='')
    if Path(os.path.join(workDir, modelFile)).is_file():
        cl = MLPR_Classifier(clases, nnShape)
        cl.load(os.path.join(workDir, modelFile))
        print('Ok.')
    else:
        sp = Sampler(workDir, trainingShp, bands)
        sp.getTrainingData()
        sp.splitData(testSize)

        cl = MLPR_Classifier(sp.classes, nnShape)
        cl.training(sp.train_inputs, sp.train_outputs)   

        mse = cl.test(sp.test_inputs, sp.test_outputs)     
        print('Ok. MSE:', round(mse, 4))

        print('Saving model ... ', end='')
        cl.save(os.path.join(workDir, modelFile))
        print('Ok.')

    zero = cl.predict([[0,0,0,0]])
    print(zero)
    if zero < 0.1:
        cl.classify(workDir, tifFile, bands)

if __name__ == '__main__':
    main()