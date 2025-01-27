#Script para regerar os conjuntos de treinamento/validação/test

import os
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
from colorama import Fore

input_folder = 'C:/Users/vntrolp/breast_density_classification/mias_cropped/cropped_path'
base_folder = 'C:/Users/vntrolp/breast_density_classification/mias_cropped'

dense_classes = ['0','1','2']
set_types = ['train','valid','test']

for cls in dense_classes:
    for st in set_types:
        output_folder = os.path.sep.join([base_folder, 'new_'+st+'_path_cropped', cls])
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)

test_size=0.3

image_names = []
image_classes = []

for cls in dense_classes:
    for f in os.listdir( input_folder+'/'+cls ):
        image_classes.append(cls)
        image_names.append(f)

input_train, input_val, label_train, label_val = train_test_split(image_names, image_classes, test_size=test_size)
input_test, input_val, label_test, label_val = train_test_split(input_val, label_val, test_size=0.5)

for i in tqdm(range(0, len(input_train)), desc='create train set:', position=0, 
                leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET)):
    origfilepath = os.path.sep.join([input_folder, label_train[i], input_train[i]])
    newfilepath = os.path.sep.join([base_folder, 'new_train_path_cropped', label_train[i], input_train[i]])
    shutil.copy2(origfilepath, newfilepath)

for i in tqdm(range(0, len(input_val)), desc='create valid set:', position=0, 
                leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET)):
    origfilepath = os.path.sep.join([input_folder, label_val[i], input_val[i]])
    newfilepath = os.path.sep.join([base_folder, 'new_valid_path_cropped', label_val[i], input_val[i]])
    shutil.copy2(origfilepath, newfilepath)

for i in tqdm(range(0, len(input_test)), desc='create test set:', position=0, 
                leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET)):
    origfilepath = os.path.sep.join([input_folder, label_test[i], input_test[i]])
    newfilepath = os.path.sep.join([base_folder, 'new_test_path_cropped', label_test[i], input_test[i]])
    shutil.copy2(origfilepath, newfilepath)