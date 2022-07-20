
import argparse  # para pasar argumentos
import os
import yaml
import time
import cv2
import gc
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

# =============================================================================
# MODELS DEFINITION
# =============================================================================
modelsDict = {'vgg16': VGG16(weights='imagenet', include_top=False), 'vgg19': VGG19(weights='imagenet', include_top=False), 'resnet50': ResNet50(
    weights='imagenet', include_top=False), 'resnet152': ResNet152(
    weights='imagenet', include_top=False), 'edrvfl': EnsembleDeepRVFL()}
# =============================================================================
# =============================================================================


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

def extract_features(filename, folder_path):
    print('\n\nEXTRACTING DATA \n\n')
    allImages = glob.glob(folder_path+'/*/*.jpg')
    images = []  # Lista donde se almacenan los features de cada imagen
    for x in range(2):
        img=allImages[x]
        img = image.load_img(img, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        images.append(x)
        print(x.shape)

    flattened = []
    model = modelsDict[nomModel]
    for x in images:
        features = model.predict(x)
        featuresFlattened = np.ravel(features)
        flattened.append(featuresFlattened)

    # numpy array con los features que queremos guardar en el fichero pkl
    flattened_arr = np.asarray(flattened)
    print(flattened_arr.shape)
    with open(filename, 'wb') as fid:
        pkl.dump(flattened_arr, fid)
    print('\n\nDATA EXTRACTED\n\n')
# =============================================================================
# PARAMETERS
# =============================================================================


sizeTilesH = 20
sizeTilesV = 20
threshold = 0.25
nomModel='resnet50'
modelFile='trained_model_' + nomModel + '.pkl'

# =============================================================================
# READING DATA
# =============================================================================
def obtainData(annotationsFile,imgPath):
    allImages = glob.glob(imgPath)
    allTiles=list()
    allLabels=list()
    #eliminar esto
    prueba=[]
    for img in sorted(allImages):
        prueba.append(img)

    #hasta aquí
    
    #for img in sorted(allImages):
    for x in range(100):
        img=prueba[x]
        image = cv2.imread(img)
        nombre=img.split('images/')[1]
        #print(nombre)
        annotations = {}
        fid = open(annotationsFile,'r')

        indexLine = 0

        for line in fid:
            indexLine = indexLine + 1
            if indexLine>1:
                lineSplit = line.strip().split(',')
                annotations[lineSplit[0]] = {'width':-1,'height':-1,'class':-1,'xmin':-1,'ymin':-1,'xmax':-1,'ymax':-1}
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

        image_tiles, offset_H, offset_V = divide_img_blocks_per_size(image,(sizeTilesH,sizeTilesV))
        annotations_tiles, mask = process_and_annotate_tiles(image_tiles,{'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax},(offset_H,offset_V))

        # =============================================================================
        # PLOTTING
        # =============================================================================
        tile = list()
        labelTile = list()

        for i in range(image_tiles.shape[1]):
            for j in range(image_tiles.shape[0]):
                tile.append(image_tiles[j][i])
                labelTile.append(annotations_tiles[j][i])

        tile_array=(np.asarray(tile))
        label_array=(np.asarray(labelTile))
        allTiles.append((tile_array,image_tiles.shape))
        allLabels.append((label_array))

    allTiles=np.asarray(allTiles,dtype=object)
    allLabels=np.asarray(allLabels,dtype=object)

    return allTiles,allLabels

def extract_tiles_features(filename, data):
    print('\n\nEXTRACTING DATA \n\n')
    tiles = []  # Lista donde se almacenan los features de cada imagen
    
    for y in data:
        
        for x in y[0]:
            #print(x)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            tiles.append(x)
            #print(x.shape)
    print(len(tiles))
    flattened = []
    model = modelsDict[nomModel]
    for x in tiles:
        features = model.predict(x)
        featuresFlattened = np.ravel(features)
        flattened.append(featuresFlattened)
    gc.collect()
    # numpy array con los features que queremos guardar en el fichero pkl
    flattened_arr = np.asarray(flattened)
    #print(flattened_arr.shape)
    with open(filename, 'wb') as fid:
        pkl.dump(flattened_arr, fid)
    print('\n\nDATA EXTRACTED\n\n')

def saveLabels(filename,data):
    labels=[]
    for y in data:
        '''for x in(y[0]):'''
        labels.append(y)
    
    with open(filename, 'wb') as fid:
        pkl.dump(labels, fid)

def unpickle(file):
    with open(file, "rb") as fid:
        data = pkl.load(fid, encoding="bytes")
        data_np = np.asarray(data,dtype=object)

    return data_np

# =============================================================================
# TRAINNING  DATA   
# =============================================================================
def prepare_data(annotations_file,img_path,pkl_data_file,pkl_labels_file):
    [Tiles, Labels]=obtainData(annotations_file,img_path)
    start = time.time()
    dims=obtainDims(Tiles)
    extract_tiles_features(pkl_data_file,Tiles)
    gc.collect()
    stop = time.time()
    print("Features Extracting time: %.3fs" % (stop - start))
    saveLabels(pkl_labels_file,Labels)
    return dims

def obtainLabels(labels_array):
    labels=[]
    
    for x in labels_array:
        for y in x:
            labels.append(y)
    
    return labels


def obtainDims(tiles_array):
    dims=[]
    for x in tiles_array:
        dims.append(x[1])
    return dims


def obtainTiles(tiles_array):
    tiles=[]
    for x in range(len(tiles_array)):
        for y in tiles_array[x][0]:
            tiles.append(y)
    return tiles


def train(model, data, labels_array):
    # Obtenemos las labels de cada imagen para entrenar
    labels = obtainLabels(labels_array)
    labels = np.asarray(labels)
    # FIT
    print('\n\nTRAINING\n\n')
    start = time.time()
    model.fit((data), labels)  # Entrenamos el modelo
    stop = time.time()
    print("Training time: %.3fs" % (stop - start))
    gc.collect()
    print('\n\nMODEL TRAINED\n\n')

def detect(model, data,labels_array,dims):
    y_pred = model.predict(data)
    l=obtainLabels(labels_array)
    p=list()
    for i in range(len(labels_array)):
        a=labels_array[i]
        a=a.reshape(dims[i])
        p.append(a)
    
    for i in range(len(p)):
        print(p[i])
    return y_pred


def evaluate(labels_array, y_pred):
    real_labels = np.asarray(obtainLabels(labels_array))
    score = skl.accuracy_score(real_labels, y_pred)
    print("Accuracy: %.3f" % (score))

def saveDims(filename,dims):
    with open(filename, 'wb') as fid:
        pkl.dump(dims, fid)

def main():
    train_annotations_file = '../train_dataset.csv'
    train_img_path='../animal dataset/train/*/*.jpg'
    train_pkl_data='train_data.pkl'
    train_pkl_labels='train_labels.pkl'


    test_annotations_file = '../test_dataset.csv'
    test_img_path='../animal dataset/test/*/*.jpg'
    test_pkl_data='test_data.pkl'
    test_pkl_labels='test_labels.pkl'
        #=========================
        #Descomentar para codigo completo
        #=========================
    prepare_data(train_annotations_file,train_img_path,train_pkl_data,train_pkl_labels)
    gc.collect()#############
    test_dims=prepare_data(test_annotations_file,test_img_path,test_pkl_data,test_pkl_labels)
    #==============
    # FITTING MODEL
    #==============
    train_labels=unpickle('train_labels.pkl')
    gc.collect()
    train_tiles=unpickle('train_data.pkl')
    gc.collect()#############

    model = Pipeline([('scaler', StandardScaler()), ('edrvfl', modelsDict['edrvfl'])])# Cargamos el modelo
    train(model, train_tiles, train_labels)  # Entrenamos el modelo
    gc.collect()
    pkl.dump(model, open(modelFile, 'wb'))  # guardamos el modelo en disco
    
    #saveDims('dims.pkl',test_dims)
    test_dims=unpickle('dims.pkl')
    loaded_model = pkl.load(open(modelFile, 'rb'))
    gc.collect()
    test_labels=unpickle('test_labels.pkl')
    test_tiles=unpickle('test_data.pkl')
    y_pred=detect(loaded_model,test_tiles,test_labels,test_dims)
    evaluate(test_labels,y_pred)

    
    
    
        


main()
