import os 
from src.MobileNet_V3 import build_mobilenet_v3
from keras.applications import Xception, VGG16, ResNet50, InceptionV3
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
from utils import plotConfMatrix, confusion_matrix_metrics

MODELS = {
	"vgg16": VGG16,
	"inception": InceptionV3,
	"xception": Xception, # TensorFlow ONLY
	"resnet50": ResNet50
    #"efficientnet": EfficientNet,
    #"nasnetlarge": NASNetLarge,
    #"nasnetmobile": NASNetMobile
}

TRAGETSIZE = {
    "xception": 299,
    "MobileNet_V3": 224,
    "vgg16": 224,
    "resnet50": 224,
    "inception": 299,
}

LABEL_DICT = {'Fatty': 0, 'Fatty-glandular': 1, 'Dense': 2}

def _main(args):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ** get configuration
    config_file = args.conf
    config_client = configparser.ConfigParser()
    config_client.read(config_file)

    model_weights_path = args.weight

    os.environ['CUDA_VISIBLE_DEVICES'] = config_client.get('gpu', 'gpu')

    model_name = config_client.get('model', 'model_name')
    input_size = config_client.getint('model', 'input_size')
    num_classes = config_client.getint('model', 'num_classes')
    loss_func = config_client.get('train', 'loss')
    ft_optimizer = config_client.get('train', 'ft_optimizer')
    test_directory = config_client.get('data', 'test')
    pre_processing = config_client.get('train', 'pre_processing')

    model_dim = TRAGETSIZE[model_name]
    model_instance = MODELS[model_name]

    test_folder = os.path.join(ROOT_DIR, 'test_'+model_name+'_'+ datetime.now().strftime('%Y-%m-%d_%H-%M'))

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    base_model = model_instance(input_shape=(model_dim, model_dim, 3), weights='imagenet', include_top=False)

    if model_name == 'MobileNet_V3':
        model = build_mobilenet_v3(model_dim, 3, 'small', 'avg')
        #model.compile(optimizer='adadelta', loss=loss_func, metrics=['accuracy'])
        #model.load_weights('D:/DOUTORADO_2020/breast_density_classification/weights/ep063-weights1-model-size-small-aug0.0-base-all_dataset_cropped-val_loss0.386.h5')
    elif model_name == 'xception':
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(base_model.input, predictions)
        #model.compile(optimizer='nadam', loss=loss_func, metrics=['accuracy'])
        #model.load_weights('C:/Users/vntrolp/breast_density_classification/weights/xception1-aug0.0-base-all_dataset_cropped_final.h5')
        #model.load_weights(model_weights_path)
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
    
    model.compile(optimizer=ft_optimizer, loss=loss_func, metrics=['accuracy'])
    model.load_weights(model_weights_path)

    images = []
    #images_path = 'C:/Users/vntrolp/breast_density_classification/all_dataset_cropped/test_path_cropped'
    images_path = test_directory
    subpaths = [x[0] for x in os.walk(images_path)]

    gt = []
    for subpath in subpaths[1:]:
        imgs = os.listdir(subpath)
        gt = gt + [int(os.path.basename(subpath).split('/')[-1])]*len(imgs)
        if pre_processing == 'scale':
            for img in tqdm(imgs, desc='test images with scale preprocessing', position=0, leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET)):
                img = os.path.join(subpath, img)
                img = image.load_img(img, target_size=(model_dim, model_dim))
                img = image.img_to_array(img)
                img = img / 255
                img = np.expand_dims(img, axis=0)
                images.append(img)
        elif pre_processing == 'norm':
            for img in tqdm(imgs, desc='test images with norm preprocessing', position=0, leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET)):
                img = os.path.join(subpath, img)
                img = image.load_img(img, target_size=(model_dim, model_dim))
                img = image.img_to_array(img)
                img = img / 255
                img = np.expand_dims(img, axis=0)
                images.append(img)
        elif pre_processing == 'imagenet_mean':
            for img in tqdm(imgs, desc='test images with imagenet_mean preprocessing', position=0, leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET)):
                img = os.path.join(subpath, img)
                #img = image.load_img(img, target_size=(model_dim, model_dim))
                img = cv2.imread(img)
                img = cv2.resize(img, (model_dim, model_dim))
                img = img.astype(np.float32)
                img[:, :, 0] -= 103.939
                img[:, :, 1] -= 116.779
                img[:, :, 2] -= 123.68
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                images.append(img)
        else:
            for img in tqdm(imgs, desc='test images without preprocessing', position=0, leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET)):
                img = os.path.join(subpath, img)
                img = image.load_img(img, target_size=(model_dim, model_dim))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                images.append(img)


    images = np.vstack(images)
    classes = model.predict(images)
    pred = np.argmax(classes,axis=1)
    labels = np.unique(gt)
    conf_matrix = confusion_matrix(gt, pred, labels)
    #print(conf_matrix)
    plotConfMatrix(conf_matrix, list(LABEL_DICT.keys()), os.path.join(test_folder, 'confusion_matrix.png'))

    csv_df = pd.DataFrame()
    csv_df['labels'] = list(LABEL_DICT.keys())

    acc = conf_matrix.diagonal()/conf_matrix.sum(axis=1)
    acc = [0 if math.isnan(x) else x for x in acc]
    #print(acc)
    csv_df['acc'] = acc

    jaccard = jaccard_score(gt, pred, labels, average=None)
    #print(jaccard)
    csv_df['jaccard'] = jaccard

    dice = f1_score(gt, pred, labels, average=None)
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