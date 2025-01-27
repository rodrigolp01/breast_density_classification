import os 
from src.MobileNet_V3 import build_mobilenet_v3
from keras.preprocessing import image
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model = build_mobilenet_v3(224, 3, 'large', 'avg')
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('D:/DOUTORADO_2020/MobileNetV3_keras/weights/ep049-loss0.365.h5')

image_file = 'D:/DOUTORADO_2020/MobileNetV3_keras/test_path/mdb002.png'
img = image.load_img(image_file, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255
img_batch = np.expand_dims(img_array, axis=0)

pr = model.predict(img_batch)