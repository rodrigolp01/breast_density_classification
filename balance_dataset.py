from data_aug import do_augmentation
import argparse
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def balance_dataset(images_path,output_path,class_ids):
    class_ids = class_ids.split(',')
    num_classes = len(class_ids)
    weights = [0 for i in class_ids]
    class_dict = dict(zip(class_ids,weights))
    datagen = ImageDataGenerator()
    generator = datagen.flow_from_directory(images_path)
    _, classes_counts = np.unique(generator.labels, return_counts=True)
    max_counts = max(classes_counts)
    for cls in range(num_classes):
        class_dict[str(cls)] = round(max_counts/classes_counts[cls])

    print('class multiplyers: ', class_dict)
    
    for cl in class_ids:
        images_input_path = images_path + '/' + cl
        images_output_path = output_path + '/' + cl
        num_times = class_dict[cl]
        if num_times > 2:
            num_times = num_times - 1
        do_augmentation(images_input_path, images_output_path, num_times)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Script for class balance')
    argparser.add_argument('-i', '--images_path', help='path to images')
    argparser.add_argument('-o', '--output_path', help='output path')
    argparser.add_argument('-c', '--class_ids', help='ID das classes')
    args = argparser.parse_args()
    images_path = args.images_path
    output_path = args.output_path
    class_ids = args.class_ids
    balance_dataset(images_path,output_path,class_ids)