import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import cv2
from tqdm import tqdm
from colorama import Fore


def split_dataset(dataset, orig_dataset_path, info_file_path, train_dataset_path, valid_dataset_path, test_dataset_path, map_classes, test_size):

    for c in map_classes.values():
        train_folder = os.path.sep.join([train_dataset_path, c])
        if not os.path.isdir( train_folder ):
            os.makedirs(train_folder)
        valid_folder = os.path.sep.join([valid_dataset_path, c])
        if not os.path.isdir( valid_folder ):
            os.makedirs(valid_folder)
        test_folder = os.path.sep.join([test_dataset_path, c])
        if not os.path.isdir( test_folder ):
            os.makedirs(test_folder)

    if dataset == 'mias':
        df = pd.read_csv(info_file_path, sep=" ")
        image_names, image_indexes = np.unique(df.values[:,0], return_index=True)
        densities = df.values[:,1][image_indexes]
        density_classes = [map_classes[d] for d in densities]
    elif dataset == 'inbreast':
        df = pd.read_csv(info_file_path, sep=";")
        image_numbers = list(df.values[:,5])
        densities_code = list(df.values[:,6])
        densities_code[181] = '1'
        image_names = []
        density_classes = []
        images = os.listdir(orig_dataset_path)
        for image_name in images:
            prefix = image_name.split('_')[0]
            idx = image_numbers.index(int(prefix))
            image_names.append(image_name.split('.')[0])
            if int(densities_code[idx]) > 2:
                density_classes.append(2)
            else:
                density_classes.append(int(densities_code[idx])-1)
    else:
        df = pd.read_csv(info_file_path, sep=",")
        densities_code = df.values[:,6]
        densities_code[densities_code == 4] = 3
        image_path = list(df.values[:,0])
        image_names = []
        density_classes = []
        for idx in range(0, len(image_path)):
            if densities_code[idx] != 0:
                density_classes.append(densities_code[idx]-1)
                image_names.append(image_path[idx].replace('\\', '/'))


    input_train, input_val, label_train, label_val = train_test_split(image_names, density_classes, test_size=test_size)
    input_test, input_val, label_test, label_val = train_test_split(input_val, label_val, test_size=0.5)

    for i in tqdm(range(0, len(input_train)), desc='create train set:', position=0, 
                leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET)):
        f = input_train[i] + '.png' if dataset != 'mini_ddsm' else input_train[i]
        c = str(label_train[i])
        origfilepath = os.path.sep.join([orig_dataset_path, f])
        if dataset == 'mini_ddsm':
            f = f.split('/')[-1]
        newfilepath = os.path.sep.join([train_dataset_path, c, f])
        shutil.copy2(origfilepath, newfilepath)
        imgredimension(newfilepath, 224, 224, 'left')

    for i in tqdm(range(0, len(input_val)), desc='create valid set:', position=0, 
                leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET)):
        f = input_val[i] + '.png' if dataset != 'mini_ddsm' else input_val[i]
        c = str(label_val[i])
        origfilepath = os.path.sep.join([orig_dataset_path, f])
        if dataset == 'mini_ddsm':
            f = f.split('/')[-1]
        newfilepath = os.path.sep.join([valid_dataset_path, c, f])
        shutil.copy2(origfilepath, newfilepath)
        imgredimension(newfilepath, 224, 224, 'left')

    for i in tqdm(range(0, len(input_test)), desc='create test set:', position=0, 
                leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET)):
        f = input_test[i] + '.png' if dataset != 'mini_ddsm' else input_test[i]
        c = str(label_test[i])
        origfilepath = os.path.sep.join([orig_dataset_path, f])
        if dataset == 'mini_ddsm':
            f = f.split('/')[-1]
        newfilepath = os.path.sep.join([test_dataset_path,c , f])
        shutil.copy2(origfilepath, newfilepath)
        imgredimension(newfilepath, 224, 224, 'left')

def imgredimension(imagepath, width, height, norm_pos=None):
    img = cv2.imread(imagepath)
    dim = (width, height)
    img_resize = cv2.resize(img, dim)
    if norm_pos:
        img_resize, _ = flip_image(img_resize, norm_pos)
    cv2.imwrite(imagepath, img_resize)

def flip_image(image, new_pos='left'):
    flipped = False
    image_pos = check_image_orientation(image)
    if image_pos != new_pos:
        image = cv2.flip(image, 1)
        flipped = True
    return image, flipped

def check_image_orientation(image):
    img_orientation = None
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_sum = np.sum(img, axis=0)
    img_mean = int(len(img_sum)/2)
    left_sum = np.sum(img_sum[0:img_mean])
    right_sum = np.sum(img_sum[img_mean::])
    if left_sum > right_sum:
        img_orientation = 'left'
    else:
        img_orientation = 'right'
    return img_orientation

if __name__ == '__main__':
    #dataset = "mias" #mias or inbreast
    #orig_dataset_path = "D:/DOUTORADO_2020/detection_mias/DATASETS/MIAS_PNG/MIAS/images"
    #info_file_path = "D:/DOUTORADO_2020/mias-mammography/Info.txt"

    dataset = "inbreast" #mias or inbreast
    orig_dataset_path = "D:/DOUTORADO_2020/detection_mias/DATASETS/INBREAST_FULL_VIEW_PNG/images"
    info_file_path = "D:/DOUTORADO_2020/INbreast_Release_1_0/INbreast_Release_1_0/INbreast.csv"

    #dataset = "mini_ddsm" #mias or inbreast
    #orig_dataset_path = "D:/DOUTORADO_2020/mini_ddsm/MINI-DDSM-Complete-PNG-16"
    #info_file_path = "D:/DOUTORADO_2020/mini_ddsm/Data-MoreThanTwoMasks/Data-MoreThanTwoMasksCsv.csv"

    train_dataset_path = "D:/DOUTORADO_2020/breast_density_classification/inbreast/train_path"
    valid_dataset_path = "D:/DOUTORADO_2020/breast_density_classification/inbreast/valid_path"
    test_dataset_path = "D:/DOUTORADO_2020/breast_density_classification/inbreast/test_path"
    map_classes = {'F': '0', 'G': '1', 'D': '2'}
    test_size = 0.2

    split_dataset(dataset, orig_dataset_path, info_file_path, train_dataset_path, valid_dataset_path, test_dataset_path, map_classes, test_size)