#from PIL import Image
#from IPython.display import display
import numpy as np
import tensorflow as tf
import keras
from keras.applications import mobilenet_v2
import eli5


model = mobilenet_v2.MobileNetV2(include_top=True, weights='imagenet', classes=1000)
image_uri = 'D:/DOUTORADO_2020/breast_density_classification/mias/test_path/0/mdb026.png'
dims = model.input_shape[1:3]
im = keras.preprocessing.image.load_img(image_uri, target_size=dims)
doc = keras.preprocessing.image.img_to_array(im)
doc = np.expand_dims(doc, axis=0)
mobilenet_v2.preprocess_input(doc)
#predictions = model.predict(doc)
model.summary()
r=eli5.show_prediction(model, doc, image=im, layer='Conv_1')
r.save('eli5.png')