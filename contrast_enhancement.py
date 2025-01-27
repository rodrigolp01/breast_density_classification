#Script for contrast enhancement before training

import cv2
import argparse
import os
import shutil


def do_contrast_enhancement(images_path, output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(images_path):
        img = cv2.imread(os.path.sep.join([images_path, filename]))
        #-----Converting image to LAB Color model----------------------------------- 
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        #-----Splitting the LAB image to different channels-------------------------
        l, a, b = cv2.split(lab)
        #-----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl,a,b))
        #-----Converting image from LAB Color model to RGB model--------------------
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        cv2.imwrite(os.path.sep.join([output_path, filename]),final)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Script for contrast enhancement before training')
    argparser.add_argument('-i', '--images_path', help='path to images')
    argparser.add_argument('-o', '--output_path', help='output path')
    args = argparser.parse_args()
    images_path = args.images_path
    output_path = args.output_path
    do_contrast_enhancement(images_path, output_path)