import os 
from src.MobileNet_V3 import build_mobilenet_v3
from keras.preprocessing import image
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, classification_report
import math
from src.generator import DataGenerator
from keras.preprocessing.image import ImageDataGenerator

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model = build_mobilenet_v3(224, 3, 'large', 'avg')
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('D:\DOUTORADO_2020\MobileNetV3_keras\weights\ep043-loss0.224.h5')

images = []
idx = 2
images_path = 'D:/DOUTORADO_2020/MobileNetV3_keras/test_path/' + str(idx)

for img in os.listdir(images_path):
    img = os.path.join(images_path, img)
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    images.append(img)

images = np.vstack(images)
classes = model.predict(images)
#print(np.argmax(classes,axis=1))
gt0 = [0]*108
gt1 = [1]*348
gt2 = [2]*397
gt = [gt0,gt1,gt2][idx]
conf_matrix = confusion_matrix(gt, np.argmax(classes,axis=1), [0,1,2])
print(conf_matrix)
acc = conf_matrix.diagonal()/conf_matrix.sum(axis=1)
acc = [0 if math.isnan(x) else x for x in acc]
print(acc)
#print(precision_score(gt, np.argmax(classes,axis=1), [0,1,2], average=None))

#'0'
#[[82 21  5]
# [ 0  0  0]
# [ 0  0  0]]
#[0.7592592592592593, 0, 0]

#'1'
#[[  0   0   0]
# [ 20 301  27]
# [  0   0   0]]
#[0, 0.8649425287356322, 0]

#'2'
#[[  0   0   0]
# [  0   0   0]
# [  1  16 380]]
#[0, 0, 0.9571788413098237]

#test_directory = 'D:/DOUTORADO_2020/MobileNetV3_keras/test_path'
# batch_size = 1
# steps = math.floor(853/batch_size)
# test_generator = DataGenerator(dir_path=test_directory, batch_size=batch_size, aug_freq=0, image_size=224)
# pred = model.predict_generator(test_generator, verbose=1, steps=steps)
# pred = np.argmax(pred, axis=1)
# print('pred: ', pred)
# classes = map(int, test_generator.label_list)
# conf_matrix = confusion_matrix(list(classes), pred, [0,1,2])
# print(conf_matrix)
# print(classification_report(list(classes), pred, target_names=['0','1','2']))

# test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_directory(test_directory,target_size=(224, 224),batch_size=1,class_mode='categorical')
# steps = math.floor(853/1)
# pred = model.predict_generator(test_generator, verbose=1, steps=steps)
# pred = np.argmax(pred, axis=1)
# print(confusion_matrix(test_generator.classes, pred))
# print(classification_report(test_generator.classes, pred, target_names=['0','1','2']))