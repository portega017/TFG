import glob
import numpy as np
from tqdm import tqdm
import cv2
from tiler import Tiler, Merger
from matplotlib import pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_input_resnet50
from keras.applications.vgg16 import VGG16, preprocess_input as preprocess_input_VGG16
from keras.applications.vgg19 import VGG19, preprocess_input as preprocess_input_VGG19
from keras.applications.mobilenet import MobileNet, preprocess_input as preprocess_input_MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as preprocess_input_MobileNet_v2
from keras.applications.mobilenet_v3 import MobileNetV3Small, preprocess_input as preprocess_input_MobileNet_v3_small
from keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input as preprocess_input_EfficientNetV2B0
from keras.applications.nasnet import NASNetMobile, preprocess_input as preprocess_input_NASNet_mobile
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_input_inception_v3

import gc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from EnsembleDeepRVFL import EnsembleDeepRVFL
import pickle
from sklearn.decomposition import PCA

# =============================================================================
# PARAMETERS
# =============================================================================

TILE_SIZE_H = 32
TILE_SIZE_V = 32
OVERLAP = 0.15
N_COMPONENTS = 100
NUM_IMAGES_TRAIN = 300
NUM_IMAGES_TEST = 36
THRESHOLD = 0.5
CLASSIFIER = 'RVFL'
FEATUREEXTRACTOR = 'NASNetMobile'

train_annotations_file = './train_dataset.csv'
train_img_path='./animal dataset/train/*/*.jpg'
file_train_data_dumped = './TRAIN_DATA_DUMPED_'+FEATUREEXTRACTOR+'.pkl'

test_annotations_file = './test_dataset.csv'
test_img_path='./animal dataset/test/*/*.jpg'
file_test_data_dumped = './TEST_DATA_DUMPED_'+FEATUREEXTRACTOR+'.pkl'

LOAD_FROM_DISK = False

# =============================================================================
# FUNCTIONS
# =============================================================================

featureExtractors = {'vgg16':{'model':VGG16(weights='imagenet', include_top=False),'preprocessor':preprocess_input_VGG16}, 
                     'vgg19':{'model':VGG19(weights='imagenet', include_top=False),'preprocessor':preprocess_input_VGG19}, 
                     'resnet50':{'model':ResNet50(weights='imagenet', include_top=False),'preprocessor':preprocess_input_resnet50},
                     'mobilenet':{'model':MobileNet(weights='imagenet', include_top=False),'preprocessor':preprocess_input_MobileNet},
                     'mobilenetv2':{'model':MobileNetV2(weights='imagenet', include_top=False),'preprocessor':preprocess_input_MobileNet_v2},
                     'mobilenetv3_small':{'model':MobileNetV3Small(weights='imagenet', include_top=False),'preprocessor':preprocess_input_MobileNet_v3_small},
                     'efficientNetV2B0':{'model':EfficientNetV2B0(weights='imagenet', include_top=False),'preprocessor':preprocess_input_EfficientNetV2B0},
                     'NASNetMobile':{'model':NASNetMobile(input_shape=(224, 224, 3),weights='imagenet', include_top=False),'preprocessor':preprocess_input_NASNet_mobile},
                     'inception_v3':{'model':InceptionV3(weights='imagenet', include_top=False),'preprocessor':preprocess_input_inception_v3}}

def process_and_annotate_tiles(image_tiles_ids, tiler, imgShape, boundingBox, threshold=0.25):

    xmin = boundingBox['xmin']
    xmax = boundingBox['xmax']
    ymin = boundingBox['ymin']
    ymax = boundingBox['ymax']
    
    mask = np.full((imgShape[0],imgShape[1]), 0, dtype=np.uint8)
    mask[ymin:ymax, xmin:xmax] = 1
    annotation_tiles = {}
    
    for tileID in image_tiles_ids:
        
        annotation_tiles[tileID]={}
        tileBbox_min_corner, tileBbox_max_corner = tiler.get_tile_bbox(tileID,all_corners=False)
        window = mask[tileBbox_min_corner[0]:tileBbox_max_corner[0], tileBbox_min_corner[1]:tileBbox_max_corner[1]]
        annotation_tiles[tileID]['sumOverlap']=np.sum(window)/(TILE_SIZE_H*TILE_SIZE_V)
        if annotation_tiles[tileID]['sumOverlap'] >= threshold:
            annotation_tiles[tileID]['annotation'] = 1
            annotation_tiles[tileID]['binaryTileMask'] = np.ones((TILE_SIZE_H,TILE_SIZE_V,3))
        else:
            annotation_tiles[tileID]['annotation'] = 0
            annotation_tiles[tileID]['binaryTileMask'] = np.zeros((TILE_SIZE_H,TILE_SIZE_V,3))
            
    return annotation_tiles, mask

def extract_features_from_tiles(image_tiles,nomModel):
    
    model = featureExtractors[nomModel]['model']
    preprocessor = featureExtractors[nomModel]['preprocessor']
    resizedTiles = []
        
    for tile in image_tiles:
        resizedTiles.append(cv2.resize(tile, (224, 224)))
    
    resizedTiles = np.asarray(resizedTiles)
    tileFeatures = model.predict(preprocessor(resizedTiles),verbose=0)
    
    gc.collect()
    tileFeatures = np.asarray(tileFeatures)
    return tileFeatures

# =============================================================================
# MAIN
# =============================================================================

allImagesTrain = glob.glob(train_img_path)
allImagesTest = glob.glob(test_img_path)

if LOAD_FROM_DISK:
    
    with open(file_train_data_dumped,'rb') as fid:
        train_annot = pickle.load(fid)
        
    with open(file_test_data_dumped,'rb') as fid:
        test_annot = pickle.load(fid)

else:         

    # 
    # TRAIN
    # 
        
    fid = open(train_annotations_file,'r')
    train_annot = {}
    linecount = 0
    
    for line in fid:
        if linecount>=1:
            lineSplit = line.strip().split(',')
            train_annot[lineSplit[0]] = {'width':-1,'height':-1,'class':-1,'xmin':-1,'ymin':-1,'xmax':-1,'ymax':-1}
            train_annot[lineSplit[0]]['width'] = int(lineSplit[1])
            train_annot[lineSplit[0]]['height'] = int(lineSplit[2])
            train_annot[lineSplit[0]]['class'] = lineSplit[3]
            train_annot[lineSplit[0]]['xmin'] = int(lineSplit[4])
            train_annot[lineSplit[0]]['ymin'] = int(lineSplit[5])
            train_annot[lineSplit[0]]['xmax'] = int(lineSplit[6])
            train_annot[lineSplit[0]]['ymax'] = int(lineSplit[7])
        linecount = linecount + 1
    
    tilers = []
    
    for imgCounter in tqdm(range(NUM_IMAGES_TRAIN)):
        
        image = cv2.imread(allImagesTrain[imgCounter])
        nameImageFile = allImagesTrain[imgCounter].split('\\')[-1]
        train_annot[nameImageFile]['fullImagePath']=allImagesTrain[imgCounter]
        
        myTiler = Tiler(data_shape=image.shape,
                      tile_shape=(TILE_SIZE_H, TILE_SIZE_V, 3),
                      overlap = OVERLAP,
                      constant_value = 128,
                      channel_dimension=2)
        
        myMerger = Merger(myTiler,window='bartlett')
        image_tiles = [tile for tile_id, tile in myTiler.iterate(image)]
        image_tiles_ids = [tile_id for tile_id, tile in myTiler.iterate(image)]
        
        annotations_tiles, mask = process_and_annotate_tiles(image_tiles_ids, myTiler, image.shape, {'xmin':train_annot[nameImageFile]['xmin'],
                                                                                           'xmax':train_annot[nameImageFile]['xmax'],
                                                                                           'ymin':train_annot[nameImageFile]['ymin'],
                                                                                           'ymax':train_annot[nameImageFile]['ymax']},THRESHOLD)
        
        train_annot[nameImageFile]['imageTiles'] = image_tiles
        train_annot[nameImageFile]['imageTilesID'] = image_tiles_ids
        train_annot[nameImageFile]['tiler'] = myTiler
        train_annot[nameImageFile]['merger'] = myMerger
        train_annot[nameImageFile]['annotationTiles'] = annotations_tiles
        
        featureVectors = extract_features_from_tiles(image_tiles,FEATUREEXTRACTOR)
        train_annot[nameImageFile]['featureTiles'] = featureVectors   
    
    with open(file_train_data_dumped,'wb') as fid:
        pickle.dump(train_annot,fid)

    # 
    # TEST
    # 
        
    fid = open(test_annotations_file,'r')
    test_annot = {}
    linecount = 0
    
    for line in fid:
        if linecount>=1:
            lineSplit = line.strip().split(',')
            test_annot[lineSplit[0]] = {'width':-1,'height':-1,'class':-1,'xmin':-1,'ymin':-1,'xmax':-1,'ymax':-1}
            test_annot[lineSplit[0]]['width'] = int(lineSplit[1])
            test_annot[lineSplit[0]]['height'] = int(lineSplit[2])
            test_annot[lineSplit[0]]['class'] = lineSplit[3]
            test_annot[lineSplit[0]]['xmin'] = int(lineSplit[4])
            test_annot[lineSplit[0]]['ymin'] = int(lineSplit[5])
            test_annot[lineSplit[0]]['xmax'] = int(lineSplit[6])
            test_annot[lineSplit[0]]['ymax'] = int(lineSplit[7])
        linecount = linecount + 1
    
    tilers = []
    
    for imgCounter in tqdm(range(NUM_IMAGES_TEST)):
        
        image = cv2.imread(allImagesTest[imgCounter])
        nameImageFile = allImagesTest[imgCounter].split('\\')[-1]
        test_annot[nameImageFile]['fullImagePath']=allImagesTest[imgCounter]
        
        myTiler = Tiler(data_shape=image.shape,
                      tile_shape=(TILE_SIZE_H, TILE_SIZE_V, 3),
                      overlap = OVERLAP,
                      constant_value = 128,
                      channel_dimension=2)
        
        myMerger = Merger(myTiler,window='bartlett')
        image_tiles = [tile for tile_id, tile in myTiler.iterate(image)]
        image_tiles_ids = [tile_id for tile_id, tile in myTiler.iterate(image)]
        
        annotations_tiles, mask = process_and_annotate_tiles(image_tiles_ids, myTiler, image.shape, {'xmin':test_annot[nameImageFile]['xmin'],
                                                                                           'xmax':test_annot[nameImageFile]['xmax'],
                                                                                           'ymin':test_annot[nameImageFile]['ymin'],
                                                                                           'ymax':test_annot[nameImageFile]['ymax']},THRESHOLD)
        
        test_annot[nameImageFile]['imageTiles'] = image_tiles
        test_annot[nameImageFile]['imageTilesID'] = image_tiles_ids
        test_annot[nameImageFile]['tiler'] = myTiler
        test_annot[nameImageFile]['merger'] = myMerger
        test_annot[nameImageFile]['annotationTiles'] = annotations_tiles
        
        featureVectors = extract_features_from_tiles(image_tiles,FEATUREEXTRACTOR)
        test_annot[nameImageFile]['featureTiles'] = featureVectors   
    
    with open(file_test_data_dumped,'wb') as fid:
        pickle.dump(test_annot,fid)

# 
# FIT
# 

subsetKeysTrain = [nameImageFile.split('\\')[-1] for nameImageFile in allImagesTrain[0:NUM_IMAGES_TRAIN]]

Xtrain = [train_annot[key]['featureTiles'] for key in subsetKeysTrain]
Xtrain = np.concatenate(Xtrain)
Xtrain = Xtrain.reshape(*Xtrain.shape[:-3], -1)
ytrain = np.array([train_annot[key]['annotationTiles'][tileID]['annotation'] for key in subsetKeysTrain for tileID in train_annot[key]['annotationTiles'].keys()])

subsetKeysTest = [nameImageFile.split('\\')[-1] for nameImageFile in allImagesTest[0:NUM_IMAGES_TEST]]

Xtest = [test_annot[key]['featureTiles'] for key in subsetKeysTest]
Xtest = np.concatenate(Xtest)
Xtest = Xtest.reshape(*Xtest.shape[:-3], -1)
ytest = np.array([test_annot[key]['annotationTiles'][tileID]['annotation'] for key in subsetKeysTest for tileID in test_annot[key]['annotationTiles'].keys()])

myPCA = PCA(n_components=N_COMPONENTS)
Xtrain_pca = myPCA.fit_transform(Xtrain)
Xtest_pca = myPCA.transform(Xtest)

if CLASSIFIER == 'RVFL':
    model = Pipeline([('scaler', StandardScaler()), ('edrvfl', EnsembleDeepRVFL(n_layer=4))]) 
elif CLASSIFIER == 'RF':
    from sklearn.ensemble import RandomForestClassifier
    model = Pipeline([('scaler', StandardScaler()), ('edrvfl', RandomForestClassifier(n_estimators=100))])
                      
model.fit(Xtrain_pca, ytrain)

# 
# PREDICT
#

ypred = model.predict(Xtest_pca)
ypred_proba = model.predict_proba(Xtest_pca)

#
# PLOT
#

plt.subplot(6,2*6,1)
counterTiles = 0

for i in range(NUM_IMAGES_TEST):
    
    ax = plt.subplot(6,2*6,2*i+2)
    nameImageFile = allImagesTest[i].split('\\')[-1]
    image = cv2.imread(test_annot[nameImageFile]['fullImagePath'])
    ax.imshow(image)
    ax.set_title('Estimated, image '+str(i))
    
    test_annot[nameImageFile]['merger'].reset()
    
    for tile, tile_id in zip(test_annot[nameImageFile]['imageTiles'],test_annot[nameImageFile]['imageTilesID']):
        # if ypred[counterTiles]==1:
        #     test_annot[nameImageFile]['merger'].add(tile_id, np.ones((TILE_SIZE_H,TILE_SIZE_V,3)))
        # else:
        #     test_annot[nameImageFile]['merger'].add(tile_id, np.zeros((TILE_SIZE_H,TILE_SIZE_V,3)))
        test_annot[nameImageFile]['merger'].add(tile_id, ypred_proba[counterTiles,1]*np.ones((TILE_SIZE_H,TILE_SIZE_V,3)))
        
        counterTiles = counterTiles + 1
    reconstructedDetectionImage = test_annot[nameImageFile]['merger'].merge()
    reconstructedDetectionImage = np.mean(reconstructedDetectionImage,2)
    ax.imshow(reconstructedDetectionImage.astype('float'),cmap='jet',vmin=0,vmax=1,alpha=0.7)
    ax.axis('off')
    
    ax2 = plt.subplot(6,2*6,2*i+1)
    ax2.imshow(image)
    ax2.set_title('Test image '+str(i))
    
    test_annot[nameImageFile]['merger'].reset()
    
    for tile_id in test_annot[nameImageFile]['imageTilesID']:
        test_annot[nameImageFile]['merger'].add(tile_id, test_annot[nameImageFile]['annotationTiles'][tile_id]['binaryTileMask'])
        
    reconstructedDetectionImage = test_annot[nameImageFile]['merger'].merge()
    ax2.imshow(reconstructedDetectionImage.astype('float'),alpha=0.5)
    ax2.axis('off')