#https://keras.io/guides/transfer_learning/
#https://www.kaggle.com/abnera/transfer-learning-keras-xception-cnn
#https://keras.io/guides/transfer_learning/
#https://github.com/otenim/Xception-with-Your-Own-Dataset
#https://github.com/kwotsin/TensorFlow-Xception
#https://keras.io/api/applications/#usage-examples-for-image-classification-models (fine-tuning inceptionv3)
#https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
#https://github.com/qubvel/efficientnet
#https://keras.io/api/applications/nasnet/#nasnetlarge-function

#vgg16
#https://towardsdatascience.com/fine-tuning-pre-trained-model-vgg-16-1277268c537f
#https://learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
#https://www.learndatasci.com/tutorials/hands-on-transfer-learning-keras/
from keras.layers import *
import os
import numpy as np
import argparse
import configparser
from keras.models import Model
import tensorflow as tf
from src.generator import DataGenerator
from keras.applications import Xception, VGG16, ResNet50, InceptionV3, DenseNet169, NASNetLarge
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as k
from datetime import datetime
from keras import optimizers
import pandas as pd
import csv
from utils import get_all_images


MODELS = {
	"vgg16": VGG16,
	"inception": InceptionV3,
	"xception": Xception, # TensorFlow ONLY
	"resnet50": ResNet50,
    #"efficientnet": EfficientNet,
    "nasnetlarge": NASNetLarge, #https://www.kaggle.com/ashishpatel26/beginner-tutorial-nasnet-pneumonia-detection
    'densenet169': DenseNet169 #https://www.pluralsight.com/guides/introduction-to-densenet-with-tensorflow
    #"nasnetmobile": NASNetMobile
    #redes siamesas
}

TRAGETSIZE = {
    "xception": 299,
    "vgg16": 224,
    "inception": 299,
    "resnet50": 224,
    "densenet169": 224,
    "nasnetlarge": 331
}

def _main(args):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ** get configuration
    config_file = args.conf
    config_client = configparser.ConfigParser()
    config_client.read(config_file)

    # ** set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = config_client.get('gpu', 'gpu')

    # ** Model configuration
    #input_size = config_client.getint('model', 'input_size')
    num_classes = config_client.getint('model', 'num_classes')
    model_name = config_client.get('model', 'model_name')

    # ** training configuration
    epochs = config_client.getint('train', 'epochs')
    batch_size = config_client.getint('train', 'batch_size')
    save_path = config_client.get('train', 'save_path')
    class_weights = config_client.getint('train', 'class_weights')
    loss_func = config_client.get('train', 'loss')
    aug_freq = config_client.getfloat('train', 'aug_freq')
    tl_optimizer = config_client.get('train', 'tl_optimizer')
    ft_optimizer = config_client.get('train', 'ft_optimizer')
    pre_processing = config_client.get('train', 'pre_processing')
    do_aug = config_client.getint('train', 'do_aug')
    tl_patience = config_client.getint('train', 'tl_patience')
    ft_patience = config_client.getint('train', 'ft_patience')
    fine_tuning = config_client.getint('train', 'fine_tuning')
    fine_tuning_all = config_client.getint('train', 'fine_tuning_all')

    # ** Dataset 
    train_directory = config_client.get('data', 'train')
    valid_directory = config_client.get('data', 'valid')
    dataset = config_client.get('data', 'dataset')
    class_names = config_client.get('data', 'class_names')

    # ** initialize data generators
    #train_generator = DataGenerator(dir_path=train_directory, batch_size=batch_size, aug_freq=aug_freq, image_size=input_size)
    #valid_generator = DataGenerator(dir_path=valid_directory, batch_size=batch_size, aug_freq=0, image_size=input_size)
    #featurewise_center=True, featurewise_std_normalization=True

    #https://machinelearningmastery.com/how-to-normalize-center-and-standardize-images-with-the-imagedatagenerator-in-keras/
    #https://stackoverflow.com/questions/46705600/fit-image-augmentations-to-training-data-using-flow-from-directory

    if pre_processing == 'scale':
        print('scaling pre-processing images')
        validation_datagen = ImageDataGenerator(rescale=1. / 255)
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        if do_aug == 1:
            train_datagen = ImageDataGenerator(rescale=1. / 255, vertical_flip=True, rotation_range=20, brightness_range=[0.2,1.0], fill_mode="nearest")
    elif pre_processing == 'norm':
        print('normalize pre-processing images')
        validation_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
        train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
        if do_aug == 1:
            train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, vertical_flip=True, rotation_range=20, brightness_range=[0.2,1.0], fill_mode="nearest")
    elif pre_processing == 'scale_norm':
        print('normalize and scale pre-processing images')
        validation_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rescale=1. / 255)
        train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rescale=1. / 255)
        if do_aug == 1:
            train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rescale=1. / 255, vertical_flip=True, rotation_range=20, brightness_range=[0.2,1.0], fill_mode="nearest")
    elif pre_processing == 'imagenet_mean':
        print('remove imagenet mean pre-processing images')
        validation_datagen = ImageDataGenerator()
        train_datagen = ImageDataGenerator()
        if do_aug == 1:
            train_datagen = ImageDataGenerator(vertical_flip=True, rotation_range=20, brightness_range=[0.2,1.0], fill_mode="nearest")
        mean = np.array([123.68, 116.779, 103.939], dtype="float32")
        train_datagen.mean = mean
        validation_datagen.mean = mean
    elif pre_processing == 'imagenet_mean_scaled':
        print('remove imagenet mean pre-processing images and scale')
        validation_datagen = ImageDataGenerator(rescale=1. / 255)
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        if do_aug == 1:
            train_datagen = ImageDataGenerator(rescale=1. / 255, vertical_flip=True, rotation_range=20, brightness_range=[0.2,1.0], fill_mode="nearest")
        mean = np.array([123.68, 116.779, 103.939], dtype="float32")
        train_datagen.mean = mean
        validation_datagen.mean = mean
    elif pre_processing == 'imagenet_mean_norm':
        print('remove imagenet mean pre-processing images and normalize')
        validation_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
        train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
        if do_aug == 1:
            train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, vertical_flip=True, rotation_range=20, brightness_range=[0.2,1.0], fill_mode="nearest")
        mean = np.array([123.68, 116.779, 103.939], dtype="float32")
        train_datagen.mean = mean
        validation_datagen.mean = mean
        train_datagen.std = 64.
        validation_datagen.std = 64.
    elif pre_processing == 'imagenet_mean_norm_scaled':
        print('remove imagenet mean pre-processing images, normalize and scale')
        validation_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rescale=1. / 255)
        train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rescale=1. / 255)
        if do_aug == 1:
            train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rescale=1. / 255, vertical_flip=True, rotation_range=20, brightness_range=[0.2,1.0], fill_mode="nearest")
        mean = np.array([123.68, 116.779, 103.939], dtype="float32")
        train_datagen.mean = mean
        validation_datagen.mean = mean
        train_datagen.std = 64.
        validation_datagen.std = 64.
    else:
        print('no pre-processing images')
        validation_datagen = ImageDataGenerator()
        train_datagen = ImageDataGenerator()
        if do_aug == 1:
            train_datagen = ImageDataGenerator(vertical_flip=True, rotation_range=20, brightness_range=[0.2,1.0], fill_mode="nearest")

    class_names = class_names.split(',')
    train_datagen.fit(get_all_images(train_directory,class_names))
    validation_datagen.fit(get_all_images(valid_directory,class_names))
    model_dim = TRAGETSIZE[model_name]

    if model_name == 'nasnetlarge':
        batch_size = 2
    
    train_generator = train_datagen.flow_from_directory(train_directory,target_size=(model_dim, model_dim),batch_size=batch_size,class_mode='categorical')
    valid_generator = validation_datagen.flow_from_directory(valid_directory,target_size=(model_dim, model_dim),batch_size=batch_size,class_mode='categorical')

    # ** initalize model
    model_instance = MODELS[model_name]
    base_model = model_instance(input_shape=(model_dim, model_dim, 3), weights='imagenet', include_top=False)

    if model_name == 'xception':

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(base_model.input, predictions)

        print('Classifier Layers before fine-tuning......')
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)

    elif model_name == 'vgg16':

        for layer in base_model.layers: #ou [:15] o que é melhor?
            layer.trainable = False

        print('Base Models Layers before fine-tuning......')
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name, layer.trainable)

        x = base_model.output
        x = Flatten()(x) # Flatten dimensions to for use in FC layers
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
        #x = Dense(256, activation='relu')(x)
        x = Dense(num_classes, activation='softmax')(x) # Softmax for multiclass
        model = Model(inputs=base_model.input, outputs=x)

        print('Classifier Layers before fine-tuning......')
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)

    elif model_name == 'inception':

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        print('Classifier Layers before fine-tuning......')
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)

    elif model_name == 'resnet50':

        headModel = base_model.output
        headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(256, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(num_classes, activation="softmax")(headModel)
        model = Model(inputs=base_model.input, outputs=headModel)

        for layer in base_model.layers:
	        layer.trainable = False

        print('Classifier Layers before fine-tuning......')
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)

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

        for layer in model.layers[:-8]:
            layer.trainable=False

        print('Classifier Layers before fine-tuning......')
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)

    elif model_name == 'nasnetlarge':

        for layer in base_model.layers:
	        layer.trainable = False

        x = base_model.output
        x= BatchNormalization()(x)
        x = GlobalAveragePooling2D()(x)
        x= Dropout(0.5)(x)
        x= Dense(128,activation='elu')(x)
        x= Dropout(0.5)(x) 
        preds=Dense(num_classes,activation='softmax')(x)
        model=Model(inputs=base_model.input,outputs=preds)

        print('Classifier Layers before fine-tuning......')
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)

    # Top Model Block
    #x = base_model.output
    #x = GlobalAveragePooling2D()(x)
    #predictions = Dense(num_classes, activation='softmax')(x)

    # add your top layer block to your base model
    #model = Model(base_model.input, predictions)
    #print(model.summary())

    model.compile(optimizer=tl_optimizer, loss=loss_func, metrics=['accuracy'])

    train_folder = os.path.join(ROOT_DIR, 'training', model_name, datetime.now().strftime('%Y-%m-%d_%H-%M'))

    model_filename = 'cw' + str(class_weights) + '-aug' + str(do_aug) + '-base-' + dataset + '.h5'
    weights_directory = os.path.join(train_folder, 'weights')
    if not os.path.exists(weights_directory):
        os.makedirs(weights_directory)
    csv_log_path = os.path.join(train_folder, 'logs')
    if not os.path.exists(csv_log_path):
        os.makedirs(csv_log_path)
    save_path = os.path.join(weights_directory, model_filename)
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=tl_patience, verbose=1)
    csv_log_filename = 'cw' + str(class_weights) + '-aug' + str(do_aug) + '-base-' + dataset + '.csv'
    csv_log_path = os.path.sep.join([csv_log_path, csv_log_filename])
    csvlog_callback=CSVLogger(filename=csv_log_path)

    weighing = {0: 0, 1: 0, 2: 0}
    #_, classes_counts = np.unique(train_generator.label_list, return_counts=True)
    _, classes_counts = np.unique(train_generator.labels, return_counts=True)
    total_counts = sum(classes_counts)
    for cls in range(num_classes):
        weighing[cls] = 1 - classes_counts[cls]/total_counts
    print('weights: ', weighing) #weights:  {0: 0.8456375838926175, 1: 0.6164734594264796, 2: 0.537888956680903}

    print('********************************** Training****************************')

    # ** start training
    model.fit_generator(generator       = train_generator,
                        validation_data = valid_generator,
                        epochs          = epochs/5,
                        callbacks       = [checkpoint, early_stopping, csvlog_callback],
                        class_weight    = weighing
                        )

    if fine_tuning == 1:
        model.load_weights(save_path)

        if model_name=='xception':
            based_model_last_block_layer_number = 126
        elif model_name=='inception':
            based_model_last_block_layer_number = 249
        elif model_name=='vgg16':
            based_model_last_block_layer_number = 19
        elif model_name=='resnet50':
            based_model_last_block_layer_number = 165
        elif model_name=='densenet169':
            based_model_last_block_layer_number = 371
        elif model_name=='nasnetlarge':
            based_model_last_block_layer_number = 1019

        for layer in model.layers[:based_model_last_block_layer_number]:
            layer.trainable = False
        for layer in model.layers[based_model_last_block_layer_number:]:
            layer.trainable = True

        print('Classifier Layers after fine-tuning first conv layers......')
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)

        model.compile(optimizer=ft_optimizer, loss=loss_func, metrics=['accuracy'])

        filename = 'cw' + str(class_weights) + '-aug' + str(do_aug) + '-base-' + dataset + '_finetuning.h5'

        save_path = os.path.join(weights_directory, filename)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=ft_patience, verbose=1)
        checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
        csv_log_filename = 'cw' + str(class_weights) + '-aug' + str(do_aug) + '-base-' + dataset + '_finetuning.csv'
        csv_log_path = os.path.join(train_folder, 'logs')
        csv_log_path = os.path.sep.join([csv_log_path, csv_log_filename])
        csvlog_callback=CSVLogger(filename=csv_log_path)

        print('**************************Fine-Tuning Training first conv layers****************************')

        model.fit_generator(generator       = train_generator,
                            validation_data = valid_generator,
                            epochs          = epochs,
                            callbacks       = [checkpoint, early_stopping, csvlog_callback],
                            class_weight    = weighing
                            )

    if fine_tuning_all == 1:
        #k.clear_session()
        model.load_weights(save_path)
        for layer in model.layers[0:]:
            layer.trainable = True

        print('Classifier Layers after fine-tuning all conv layers......')
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)

        model.compile(optimizer=ft_optimizer, loss=loss_func, metrics=['accuracy'])

        filename = 'cw' + str(class_weights) + '-aug' + str(do_aug) + '-base-' + dataset + '_finetuningall.h5'

        save_path = os.path.join(weights_directory, filename)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=ft_patience, verbose=1)
        checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
        csv_log_filename = 'cw' + str(class_weights) + '-aug' + str(do_aug) + '-base-' + dataset + '_finetuningall.csv'
        csv_log_path = os.path.join(train_folder, 'logs')
        csv_log_path = os.path.sep.join([csv_log_path, csv_log_filename])
        csvlog_callback=CSVLogger(filename=csv_log_path)

        print('**************************Fine-Tuning Training all conv layers****************************')

        model.fit_generator(generator       = train_generator,
                            validation_data = valid_generator,
                            epochs          = epochs,
                            callbacks       = [checkpoint, early_stopping, csvlog_callback],
                            class_weight    = weighing
                            )

    #epochs, optimizer, image pre-processing, model-artchitecture
    csv_confs = {}
    csv_confs['transfer_learning_epochs'] = epochs/5
    csv_confs['fine_tuning_epochs'] = epochs
    csv_confs['transfer_learning_optimizer'] = tl_optimizer
    csv_confs['fine_tuning_optimizer'] = ft_optimizer
    csv_confs['transfer_learning_loss_func'] = loss_func
    csv_confs['fine_tuning_loss_func'] = loss_func
    csv_confs['pre_processing'] = pre_processing #options: imagenet_mean, norm
    csv_confs['data_aug'] = do_aug
    csv_confs['class_weight'] = str(weighing)
    csv_confs['model_name'] = model_name
    csv_confs['dataset'] = dataset
    csv_confs['train_directory'] = train_directory
    csv_confs['model_path'] = save_path
    #csv_confs.to_csv(os.path.join(train_folder, 'train_configs.csv'), index=False)
    with open(os.path.join(train_folder, 'train_configs.csv'), 'w') as f:
        for key in csv_confs.keys():
            f.write("%s,%s\n"%(key,csv_confs[key]))

    with open(os.path.join('current_model.txt'), 'w') as f:
        f.write("%s"%(save_path.replace('\\','/')))

    print(save_path)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='')
    argparser.add_argument('-c', '--conf', help='path to configuration file', default='config/train.ini')   

    args = argparser.parse_args()
    _main(args)

    k.clear_session()