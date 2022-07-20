# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 12:16:50 2022

@author: Pablo Ortega
"""
import argparse  # para pasar argumentos
import os
import yaml
import time
from CustomModels import EnsembleDeepRVFL
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet import ResNet152
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import pickle as pkl
import numpy as np
import glob
import sklearn.metrics as skl
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# DATASET
# https://drive.google.com/drive/folders/1iA6XY8rhdoShDi8Bi-efkCg3LFaTzX2P?usp=sharing



def extractData(dataPath):
    with open(dataPath, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

# INPUT ARGS
nomModel='resnet50'
dataPath='./'
args = extractData(os.path.sep.join([dataPath,'data.yml']))
# END INPUT ARGS

# VARIABLES DEFINITION


testDir = args['test']
trainDir = args['train']
valDir = args['val']
modelFile = os.path.sep.join([dataPath, 'trained_model_'+nomModel+'.sav'])




pklTrain = os.path.sep.join([dataPath,'data_train_'+nomModel+'.pkl'])
pklTest = os.path.sep.join([dataPath,'data_test_'+nomModel+'.pkl'])
pklVal = os.path.sep.join([dataPath,'data_val_' + nomModel + '.pkl'])


# END VARIABLES DEFINITION

# MODELS DEFINITION
modelsDict = {'vgg16': VGG16(weights='imagenet', include_top=False), 'vgg19': VGG19(weights='imagenet', include_top=False),'resnet50': ResNet50(
    weights='imagenet', include_top=False), 'resnet152': ResNet152(
    weights='imagenet', include_top=False), 'edrvfl': EnsembleDeepRVFL()}
# END MODELS DEFINITION

def extract_features(filename, folder_path):
    print('\n\nEXTRACTING DATA \n\n')
    allImages = glob.glob(folder_path+'/*/*.jpg')
    images = []  # Lista donde se almacenan los features de cada imagen
    for img in sorted(allImages):
        img = image.load_img(img, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        images.append(x)

    flattened = []
    model = modelsDict[nomModel]
    for x in images:
        features = model.predict(x)
        featuresFlattened = np.ravel(features)
        flattened.append(featuresFlattened)

    # numpy array con los features que queremos guardar en el fichero pkl
    flattened_arr = np.asarray(flattened)

    with open(filename, 'wb') as fid:
        pkl.dump(flattened_arr, fid)
    print('\n\nDATA EXTRACTED\n\n')


def unpickle(file):
    with open(file, "rb") as fid:
        data = pkl.load(fid, encoding="bytes")
        data_np = np.asarray(data)

    return data_np


def train(model, data, path):
    # Obtenemos las labels de cada imagen para entrenar
    y = obtenerLabels(path)
    y = np.asarray(y)
    # FIT
    print('\n\nTRAINING\n\n')
    start = time.time()

    model.fit(data, y)  # Entrenamos el modelo
    stop = time.time()
    print("Training time: %.3fs" % (stop - start))

    print('\n\nMODEL TRAINED\n\n')


def detect(model, data):
    label = args['labels']
    y_pred = model.predict(data)
    for i in range(len(y_pred)):
        print(label[y_pred[i]])
    return y_pred


def evaluate(path, y_pred):
    real_labels = np.asarray(obtenerLabels(path))
    score = skl.accuracy_score(real_labels, y_pred)
    print("Accuracy: %.3f" % (score))

    


def obtenerLabels(folder_path):
    allFiles= glob.glob(folder_path+'/*/*.txt')
    labels = []
    for f in sorted(allFiles):
            with open(f, "r") as fid:
                c = int(fid.read(1))
                labels.append(c)
    return labels


def classDetectionRVFL():
    start=time.time()
    # BLOQUE TRAIN
    
    #extract_features(pklTrain, trainDir) # extraemos las features del dataset de entrenamiento
    data_train = unpickle(pklTrain)
    

    print((data_train[0].shape))
    
    model = Pipeline([('scaler', StandardScaler()), ('edrvfl', modelsDict['edrvfl'])])# Cargamos el modelo
    '''train(model, data_train, trainDir)  # Entrenamos el modelo
    pkl.dump(model, open(modelFile, 'wb'))  # guardamos el modelo en disco'''
    # FIN BLOQUE TRAIN

    # cargamos el modelo,  podemos trabajar con el para evitar el tener que entrenarlo otra vez cuando solo queremos evaluar la predicción
    loaded_model = pkl.load(open(modelFile, 'rb'))

    # BLOQUE TEST
    
    #extract_features(pklTest, testDir) # extraemos las features del dataset de prueba
    data_test = unpickle(pklTest)
    # Para trabajar con el modelo guardado, usar loaded_model en la funcion detect
    ####y_pred = detect(loaded_model, data_test)
    # FIN BLOQUE TEST

    # EVALUATE MODEL
    #evaluate(testDir, y_pred)
    stop = time.time()

    print("Total execution time: %.3fs" % (stop - start))




def boundingBoxRVFL():
    print('hola')


#classDetectionRVFL()
boundingBoxRVFL()