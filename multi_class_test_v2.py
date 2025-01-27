import os 
from src.MobileNet_V3 import build_mobilenet_v3
from keras.applications import Xception, VGG16, ResNet50, InceptionV3, NASNetLarge, DenseNet169
from keras.preprocessing import image
import numpy as np
from sklearn.metrics import confusion_matrix, jaccard_score, f1_score
import math
from keras.layers import *
from keras.models import Model
import argparse
import configparser
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from shutil import copyfile
from tqdm import tqdm
from colorama import Fore
import cv2
from keras.preprocessing.image import ImageDataGenerator
from utils import plotConfMatrix, confusion_matrix_metrics, get_all_images

MODELS = {
	"vgg16": VGG16,
	"inception": InceptionV3,
	"xception": Xception, # TensorFlow ONLY
	"resnet50": ResNet50,
    #"efficientnet": EfficientNet,
    "nasnetlarge": NASNetLarge,
    'densenet169': DenseNet169
}

TRAGETSIZE = {
    "xception": 299,
    "MobileNet_V3": 224,
    "vgg16": 224,
    "resnet50": 224,
    "inception": 299,
    "densenet169": 224,
    "nasnetlarge": 331
}

LABEL_DICT = {'Fatty': 0, 'Fatty-glandular': 1, 'Dense': 2}

def get_last_model_trained(curr_file='current_model.txt'):
    with open(curr_file) as f:
        model_path = f.read()
    return model_path

def _main(args):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ** get configuration
    config_file = args.conf
    config_client = configparser.ConfigParser()
    config_client.read(config_file)

    model_weights_path = args.weight
    if not model_weights_path:
        model_weights_path = get_last_model_trained()

    os.environ['CUDA_VISIBLE_DEVICES'] = config_client.get('gpu', 'gpu')

    model_name = config_client.get('model', 'model_name')
    input_size = config_client.getint('model', 'input_size')
    num_classes = config_client.getint('model', 'num_classes')
    loss_func = config_client.get('train', 'loss')
    ft_optimizer = config_client.get('train', 'ft_optimizer')
    test_directory = config_client.get('data', 'test')
    pre_processing = config_client.get('train', 'pre_processing')
    batch_size = config_client.getint('train', 'batch_size')
    class_names = config_client.get('data', 'class_names')

    model_dim = TRAGETSIZE[model_name]
    model_instance = MODELS[model_name]

    test_folder = os.path.join(ROOT_DIR, 'test', model_name, datetime.now().strftime('%Y-%m-%d_%H-%M'))

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    base_model = model_instance(input_shape=(model_dim, model_dim, 3), weights='imagenet', include_top=False)

    if model_name == 'MobileNet_V3':
        model = build_mobilenet_v3(model_dim, 3, 'small', 'avg')
    elif model_name == 'xception':
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(base_model.input, predictions)
    elif model_name == 'vgg16':
        x = base_model.output
        x = Flatten()(x) # Flatten dimensions to for use in FC layers
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
        x = Dense(num_classes, activation='softmax')(x) # Softmax for multiclass
        model = Model(inputs=base_model.input, outputs=x)
    elif model_name == 'inception':
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
    elif model_name == 'resnet50':
        headModel = base_model.output
        headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(256, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(num_classes, activation="softmax")(headModel)
        model = Model(inputs=base_model.input, outputs=headModel)
    elif model_name == 'densenet169':
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x= BatchNormalization()(x)
        x= Dropout(0.5)(x)
        x= Dense(1024,activation='relu')(x) 
        x= Dense(512,activation='relu')(x) 
        x= BatchNormalization()(x)
        x= Dropout(0.5)(x)
        preds=Dense(num_classes,activation='softmax')(x)
        model=Model(inputs=base_model.input,outputs=preds)
    elif model_name == 'nasnetlarge':
        x = base_model.output
        x= BatchNormalization()(x)
        x = GlobalAveragePooling2D()(x)
        x= Dropout(0.5)(x)
        x= Dense(128,activation='elu')(x)
        x= Dropout(0.5)(x) 
        preds=Dense(num_classes,activation='softmax')(x)
        model=Model(inputs=base_model.input,outputs=preds)
    
    model.compile(optimizer=ft_optimizer, loss=loss_func, metrics=['accuracy'])
    model.load_weights(model_weights_path)

    if pre_processing == 'scale':
        print('scaling pre-processing images')
        test_datagen = ImageDataGenerator(rescale=1. / 255)
    elif pre_processing == 'norm':
        print('normalize pre-processing images')
        test_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    elif pre_processing == 'scale_norm':
        print('normalize and scale pre-processing images')
        test_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rescale=1. / 255)
    elif pre_processing == 'imagenet_mean':
        print('remove imagenet mean pre-processing images')
        test_datagen = ImageDataGenerator()
        mean = np.array([123.68, 116.779, 103.939], dtype="float32")
        test_datagen.mean = mean
    elif pre_processing == 'imagenet_mean_scaled':
        print('remove imagenet mean pre-processing images and scale')
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        mean = np.array([123.68, 116.779, 103.939], dtype="float32")
        test_datagen.mean = mean
    elif pre_processing == 'imagenet_mean_norm':
        print('remove imagenet mean pre-processing images and normalize')
        test_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
        mean = np.array([123.68, 116.779, 103.939], dtype="float32")
        test_datagen.mean = mean
        test_datagen.std = 64.
    elif pre_processing == 'imagenet_mean_norm_scaled':
        print('remove imagenet mean pre-processing images, normalize and scaled')
        test_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rescale=1. / 255)
        mean = np.array([123.68, 116.779, 103.939], dtype="float32")
        test_datagen.mean = mean
        test_datagen.std = 64.
    else:
        print('no pre-processing images')
        test_datagen = ImageDataGenerator()

    model_dim = TRAGETSIZE[model_name]
    class_names = class_names.split(',')
    test_datagen.fit(get_all_images(test_directory,class_names))
    test_generator = test_datagen.flow_from_directory(test_directory,target_size=(model_dim, model_dim),color_mode="rgb",batch_size=1,shuffle = False,class_mode='categorical')
    filenames = test_generator.filenames
    nb_samples = len(filenames)
    predict = model.predict_generator(test_generator,steps = nb_samples)
    predicted_class_indices=np.argmax(predict,axis=1)
    #labels = (test_generator.class_indices)
    #labels = dict((v,k) for k,v in labels.items())
    #predictions = [labels[k] for k in predicted_class_indices]
    conf_matrix = confusion_matrix(test_generator.classes, predicted_class_indices, list(LABEL_DICT.values()))
    #print(conf_matrix)
    plotConfMatrix(conf_matrix, list(LABEL_DICT.keys()), os.path.join(test_folder, 'confusion_matrix.png'))

    csv_df = pd.DataFrame()
    csv_df['labels'] = list(LABEL_DICT.keys())

    acc = conf_matrix.diagonal()/conf_matrix.sum(axis=1)
    acc = [0 if math.isnan(x) else x for x in acc]
    #print(acc)
    csv_df['acc'] = acc

    jaccard = jaccard_score(test_generator.classes, predicted_class_indices, list(LABEL_DICT.values()), average=None)
    #print(jaccard)
    csv_df['jaccard'] = jaccard

    dice = f1_score(test_generator.classes, predicted_class_indices, list(LABEL_DICT.values()), average=None)
    #print(dice)
    csv_df['dice'] = dice

    false_positive, false_negative, true_positive, true_negative, specificity, false_positive_rate, \
    false_negative_rate = confusion_matrix_metrics(conf_matrix)

    csv_df['FPR'] = false_positive_rate

    csv_df['FNR'] = false_negative_rate

    csv_df.to_csv(os.path.join(test_folder, 'metrics.csv'), index=False)

    print(false_positive_rate)

    src = os.path.dirname(os.path.dirname(model_weights_path)) + '/train_configs.csv'
    dst = test_folder + '/train_configs.csv'
    copyfile(src, dst)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='')
    argparser.add_argument('-c', '--conf', help='path to configuration file', default='config/train.ini')   
    argparser.add_argument('-w', '--weight', help='path to model weights file')

    args = argparser.parse_args()
    _main(args)