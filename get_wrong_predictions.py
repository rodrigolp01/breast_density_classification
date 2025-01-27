import os 
from src.MobileNet_V3 import build_mobilenet_v3
from keras.preprocessing import image
import numpy as np
import shutil
from tqdm import tqdm
from colorama import Fore


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model = build_mobilenet_v3(224, 3, 'small', 'avg')
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('D:/DOUTORADO_2020/breast_density_classification/weights/ep063-weights1-model-size-small-aug0.0-base-all_dataset_cropped-val_loss0.386.h5')
images_path = 'D:/DOUTORADO_2020/breast_density_classification/all_dataset_cropped/test_path_cropped'
wrong_prediction_dst = 'D:/DOUTORADO_2020/breast_density_classification/all_dataset_cropped/wrong_predictions'

subpaths = [x[0] for x in os.walk(images_path)]
for subpath in subpaths[1:]:
    imgs = os.listdir(subpath)
    class_name = os.path.basename(subpath).split('/')[-1]
    dst_folder = os.path.sep.join([wrong_prediction_dst, class_name])
    if not os.path.isdir( dst_folder ):
        os.makedirs(dst_folder)
    gt = int(os.path.basename(subpath).split('/')[-1])
    for img_name in tqdm(imgs, desc='test imagens from folder '+str(gt), position=0, leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET)):
        img_path = os.path.join(subpath, img_name)
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = img / 255
        img = np.expand_dims(img, axis=0)
        pr = model.predict(img)
        pr = np.argmax(pr,axis=1)
        pr = pr[0]
        if pr != gt:
            newfilepath = os.path.sep.join([ dst_folder, img_name.split('.')[0]+'_'+str(pr)+'.png' ])
            shutil.copy2(img_path, newfilepath)