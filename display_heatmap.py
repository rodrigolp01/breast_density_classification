import os

from keras import layers 
from src.MobileNet_V3 import build_mobilenet_v3
from keras.preprocessing import image
import numpy as np
import keras
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import tensorflow as tf
from keras.applications import mobilenet_v2
import eli5

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model = build_mobilenet_v3(224, 3, 'small', 'avg')
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('D:/DOUTORADO_2020/breast_density_classification/weights/ep063-weights1-model-size-small-aug0.0-base-all_dataset_cropped-val_loss0.386.h5')
model.summary()
last_conv_layer_name = model.layers[-3:][0].name
#image_file = 'D:/DOUTORADO_2020/breast_density_classification/all_dataset_cropped/wrong_predictions/2/A_1899_1_1.png'
image_file = 'D:/DOUTORADO_2020/breast_density_classification/all_dataset/test_path/2/A_0212_1.LEFT_CC.png'
# remove softmax
#l = model.get_layer(index=-1) # get the last (output) layer
#l.activation = keras.activations.linear # swap activation
#model.save('tmp_model_save_rmsoftmax') # note that this creates a file of the model
#model = keras.models.load_model('tmp_model_save_rmsoftmax')

img = image.load_img(image_file, target_size=(224, 224))
img_array = image.img_to_array(img)
#img_array = img_array / 255
img_array = np.expand_dims(img_array, axis=0)
mobilenet_v2.preprocess_input(img_array)

r = eli5.show_prediction(model, img_array, image=img, layer=last_conv_layer_name)
r.save('teste.png')

expl = eli5.explain_prediction(model, img_array)
print(expl)
print((expl.targets[0].target, expl.targets[0].score, expl.targets[0].proba))
image = expl.image
heatmap = expl.targets[0].heatmap
print(heatmap)