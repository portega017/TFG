# -*- coding: utf-8 -*-
"""
Created on Wed Jul 5 17:03:24 2022

@author: Pablo Ortega
"""


from re import T
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import pickle as pkl
import time

import sklearn.metrics as skl
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# =============================================================================
# IMPORT MODELS
# =============================================================================

from CustomModels import EnsembleDeepRVFL
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet import ResNet152
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import preprocess_input

# =============================================================================
# MODELS DEFINITION
modelsDict = {'vgg16': VGG16(weights='imagenet', include_top=False), 'vgg19': VGG19(weights='imagenet', include_top=False), 'resnet50': ResNet50(
    weights='imagenet', include_top=False), 'resnet152': ResNet152(
    weights='imagenet', include_top=False), 'edrvfl': EnsembleDeepRVFL()}
# =============================================================================

def extract_features(filename,tile_array):
    print('\n\nEXTRACTING TILE DATA \n\n')
    tile = []  # Lista donde se almacenan los features de cada tile
    for x in tile_array:
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        tile.append(x)
    print(len(tile))
    flattened = []
    model = modelsDict[nomModel]
    for x in tile:
        features = model.predict(x)
        featuresFlattened = np.ravel(features)
        flattened.append(featuresFlattened)

    # numpy array con los features que queremos guardar en el fichero pkl
    flattened_arr = np.asarray(flattened)
    print(flattened_arr.shape)
    with open(filename, 'wb') as fid:
        pkl.dump(flattened_arr, fid)
    print('\n\nTILE DATA EXTRACTED\n\n')

def unpickle(file):
    with open(file, "rb") as fid:
        data = pkl.load(fid, encoding="bytes")
        data_np = np.asarray(data)

    return data_np
def divide_img_blocks_per_size(img, s_blocks=(5, 5), colorPadding=(0, 0, 0)):
    size_H = img.shape[0]
    size_V = img.shape[1]
    channels = img.shape[2]
    n_blocks_H = int(size_H/s_blocks[0])+1
    n_blocks_V = int(size_V/s_blocks[1])+1
    new_image_size_H = n_blocks_H*s_blocks[0]
    new_image_size_V = n_blocks_V*s_blocks[1]
    color = colorPadding
    result = np.full((new_image_size_H, new_image_size_V,
                     channels), color, dtype=np.uint8)

    H_center = (new_image_size_H - size_H) // 2
    V_center = (new_image_size_V - size_V) // 2

    offset_H = int(0.5*(new_image_size_H - size_H))
    offset_V = int(0.5*(new_image_size_V - size_V))

    result[H_center:H_center+size_H, V_center:V_center+size_V] = img

    tiles = np.empty((n_blocks_H, n_blocks_V), dtype=object)
    index_r = 0

    for r in range(0, result.shape[0], s_blocks[0]):
        index_c = 0
        for c in range(0, result.shape[1], s_blocks[1]):
            window = result[r:r+s_blocks[0], c:c+s_blocks[1], :]
            tiles[index_r, index_c] = window
            index_c = index_c+1

        index_r = index_r+1

    return tiles, offset_H, offset_V


def process_and_annotate_tiles(image_tiles, boundingBox, offsets=(0, 0), threshold=0.25):

    offset_H = offsets[0]
    offset_V = offsets[1]
    xmin = boundingBox['xmin']
    xmax = boundingBox['xmax']
    ymin = boundingBox['ymin']
    ymax = boundingBox['ymax']

    n_blocks_H = image_tiles.shape[0]
    n_blocks_V = image_tiles.shape[1]
    size_H = image_tiles[0, 0].shape[0]
    size_V = image_tiles[0, 0].shape[1]

    mask = np.full((n_blocks_H*size_H, n_blocks_V*size_V), 0, dtype=np.uint8)
    mask[offset_V + ymin:offset_V+ymax, offset_H+xmin:offset_H+xmax] = 1

    annotations_tiles = np.empty((n_blocks_H, n_blocks_V), dtype=int)
    index_r = 0

    for r in range(0, mask.shape[0], size_H):
        index_c = 0
        for c in range(0, mask.shape[1], size_V):
            window = mask[r:r+size_H, c:c+size_V]
            sumPixels = np.sum(window)
            if sumPixels/(size_H*size_V) >= threshold:
                annotations_tiles[index_r, index_c] = 1
            else:
                annotations_tiles[index_r, index_c] = 0
            index_c = index_c+1

        index_r = index_r+1

    return annotations_tiles, mask


def chooseBBColor(objClass):
    if objClass == 'dog':
        return (255, 0, 0)
    elif objClass == 'cat':
        return (0, 255, 0)
    else:
        return (0, 0, 255)
# =============================================================================
# PARAMETERS
# =============================================================================


sizeTilesH = 64
sizeTilesV = 64
threshold = 0.25
nomModel='resnet50'

# =============================================================================
# READING DATA
# =============================================================================

def obtenerData(allImages, annotationsFile):
    todos=list()
    for x in range(1):
        img = allImages[x]
        image = cv2.imread(img)
        nombre = img.split('images/')[1]
        print(nombre)
        annotations = {}
        fid = open(annotationsFile, 'r')

        indexLine = 0

        for line in fid:
            indexLine = indexLine + 1
            if indexLine > 1:
                lineSplit = line.strip().split(',')
                annotations[lineSplit[0]] = {'width': -1, 'height': -1, 'class': -1, 'xmin': -1, 'ymin': -1, 'xmax': -1, 'ymax': -1}
                annotations[lineSplit[0]]['width'] = int(lineSplit[1])
                annotations[lineSplit[0]]['height'] = int(lineSplit[2])
                annotations[lineSplit[0]]['class'] = lineSplit[3]
                annotations[lineSplit[0]]['xmin'] = int(lineSplit[4])
                annotations[lineSplit[0]]['ymin'] = int(lineSplit[5])
                annotations[lineSplit[0]]['xmax'] = int(lineSplit[6])
                annotations[lineSplit[0]]['ymax'] = int(lineSplit[7])

        fid.close()

        xmin = annotations[nombre]['xmin']
        xmax = annotations[nombre]['xmax']
        ymin = annotations[nombre]['ymin']
        ymax = annotations[nombre]['ymax']
        objClass = annotations[nombre]['class']

        # =============================================================================
        # TILE EXTRACTION
        # =============================================================================

        image_tiles, offset_H, offset_V = divide_img_blocks_per_size(image, (sizeTilesH, sizeTilesV))
        annotations_tiles, mask = process_and_annotate_tiles(image_tiles, {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}, (offset_H, offset_V))
        tile = list()
        labelTile = list()

        for i in range(image_tiles.shape[1]):
            for j in range(image_tiles.shape[0]):
                tile.append(image_tiles[j][i])
                labelTile.append(annotations_tiles[j][i])

        tile_array=(np.asarray(tile))
        label_array=(np.asarray(labelTile))
        print(label_array)
        nomFich=nombre[:-4]+'.pkl'
        extract_features(nomFich,tile_array)
        todos.append(tile_array)
        return nomFich, label_array, image_tiles.shape

def detect(model, data,shape):
    y_pred = model.predict(data)
    y_pred=np.asarray(y_pred)
    y_pred=y_pred.reshape(shape)
    print(y_pred)
    return y_pred


def main():
    trainImages = glob.glob('../animal dataset/train/*/*.jpg')
    trainAnnotationsFile = '../train_dataset.csv'
    nomFich,label_array,shape=obtenerData(trainImages,trainAnnotationsFile)
    data_train = unpickle(nomFich)

    
    model = Pipeline([('scaler', StandardScaler()), ('edrvfl', modelsDict['edrvfl'])])# Cargamos el modelo

    # =============================================================================
    # TRAINING
    # =============================================================================

    print('\n\nTRAINING\n\n')
    start = time.time()

    model.fit(data_train, label_array)  # Entrenamos el modelo
    stop = time.time()
    print("Training time: %.3fs" % (stop - start))

    print('\n\nMODEL TRAINED\n\n')

    testImages = glob.glob('../animal dataset/test/*/*.jpg')
    testAnnotationsFile = '../test_dataset.csv'
    nomFich,label_array,shape=obtenerData(testImages,testAnnotationsFile)
    data_test = unpickle(nomFich)

    detect(model,data_test,shape)

    # print(a.shape)
    # flattened = []
    # model = modelsDict[nomModel]
    # for x in a:
    #     features = model.predict(x)
    #     featuresFlattened = np.ravel(features)
    #     flattened.append(featuresFlattened)
    

        

    # numpy array con los features que queremos guardar en el fichero pkl
    '''flattened_arr = np.asarray(flattened)
    print(flattened_arr.shape)
'''
    
    '''tile = np.asarray(tile)
    labelTile = np.asarray(labelTile)
    print(tile.shape)
    print((labelTile.shape))'''

'''    # =============================================================================
    # PLOTTING
    # =============================================================================
    color=chooseBBColor(objClass)
    img = cv2.imread(img)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(img,objClass,(xmin,ymin-3),0,0.3,color)
    #cv2.imshow("Show",img)
    cv2.imwrite('../BoundingBox/'+nombre,img)
    #cv2.waitKey()  
'''
main()