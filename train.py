import os 
import sys 
import logging
import argparse
import configparser
import numpy as np
from keras.models import load_model
from keras.layers import *
from keras import backend as K
import tensorflow as tf
from src.generator import DataGenerator
from src.learning_rate_schedule import learning_rate_scheduler
from src.MobileNet_V3 import build_mobilenet_v3
from keras.applications import Xception, VGG16, VGG19, ResNet50, InceptionV3
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import (ModelCheckpoint, 
                             LearningRateScheduler, 
                             ReduceLROnPlateau, 
                             EarlyStopping)

logging.basicConfig(level=logging.INFO)

#https://stackoverflow.com/questions/51793737/custom-loss-function-for-u-net-in-keras-using-class-weights-class-weight-not
def weightedLoss(originalLossFunc, weightsList):
    def lossFunc(true, pred):
        axis = -1 #if channels last 
        #axis=  1 #if channels first
        #argmax returns the index of the element with the greatest value
        #done in the class axis, it returns the class index
        classSelectors = K.argmax(true, axis=axis) 
        #if your loss is sparse, use only true as classSelectors
        #considering weights are ordered by class, for each class
        #true(1) if the class index is equal to the weight index   
        classSelectors = [K.equal(tf.cast(i, tf.int32), tf.cast(classSelectors, tf.int32)) for i in range(len(weightsList))]
        #casting boolean to float for calculations  
        #each tensor in the list contains 1 where ground true class is equal to its index 
        #if you sum all these, you will get a tensor full of ones. 
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]
        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel,w in zip(classSelectors, weightsList)] 
        #sums all the selections
        #result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]
        #make sure your originalLossFunc only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        lossf = originalLossFunc(true,pred) 
        lossf = lossf * weightMultiplier
        return lossf
    return lossFunc

def _main(args):

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ** get configuration
    config_file = args.conf
    config_client = configparser.ConfigParser()
    config_client.read(config_file)

    # ** set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = config_client.get('gpu', 'gpu')

    # ** MobileNet V3 configuration
    input_size = config_client.getint('model', 'input_size')
    model_size = config_client.get('model', 'model_size')
    pooling_type = config_client.get('model', 'pooling_type')
    num_classes = config_client.getint('model', 'num_classes')

    # ** training configuration
    epochs = config_client.getint('train', 'epochs')
    batch_size = config_client.getint('train', 'batch_size')
    save_path = config_client.get('train', 'save_path')
    class_weights = config_client.getint('train', 'class_weights')
    loss_func = config_client.get('train', 'loss')
    aug_freq = config_client.getfloat('train', 'aug_freq')

    # ** Dataset 
    train_directory = config_client.get('data', 'train')
    valid_directory = config_client.get('data', 'valid')
    dataset = config_client.get('data', 'dataset')

    # ** initialize data generators
    train_generator = DataGenerator(dir_path=train_directory, batch_size=batch_size, aug_freq=aug_freq, image_size=input_size)
    valid_generator = DataGenerator(dir_path=valid_directory, batch_size=batch_size, aug_freq=0, image_size=input_size)

    # ** initalize model
    try:
        model = load_model(os.path.join(ROOT_DIR, config_client.get('train', 'pretrained_path')))
    except Exception as e:
        logging.info('Failed to load pre-trained model.')
        model = build_mobilenet_v3(input_size, num_classes, model_size, pooling_type)

    if class_weights > 0:
        #all_dataset
        #0: 1.265 -> 1-1.265/8.195
        #1: 3.143 -> 1-3.143/8.195
        #2: 3.787 -> 1-3.787/8.195
        #weighing = {0: 0.84, 1: 0.61, 2: 0.53}
        #loss_func = weightedLoss(categorical_crossentropy, weighing)
        weighing = {0: 0, 1: 0, 2: 0}
        _, classes_counts = np.unique(train_generator.label_list, return_counts=True)
        total_counts = sum(classes_counts)
        for cls in range(num_classes):
            weighing[cls] = 1 - classes_counts[cls]/total_counts
        print('weights: ', weighing)

    #model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adadelta', loss=loss_func, metrics=['accuracy'])

    # ** setup keras callback
    filename = 'ep{epoch:03d}-weights' + str(class_weights) + '-model-size-' + model_size + '-aug' + str(aug_freq) + '-base-' + dataset + '-val_loss{val_loss:.3f}.h5'
    weights_directory = os.path.join(ROOT_DIR, 'weights')
    save_path = os.path.join(weights_directory, filename)
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
    scheduler = LearningRateScheduler(learning_rate_scheduler)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=1)

    # ** start training
    model.fit_generator(generator       = train_generator,
                        validation_data = valid_generator,
                        epochs          = epochs,
                        #callbacks       = [checkpoint, scheduler],
                        callbacks       = [checkpoint, early_stopping],
                        class_weight    = weighing
                        )

    #model.save(os.path.join(ROOT_DIR, save_path))




if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='')
    argparser.add_argument('-c', '--conf', help='path to configuration file')   

    args = argparser.parse_args()
    _main(args)

