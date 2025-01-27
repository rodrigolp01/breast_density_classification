#Script for data augmentation and class balance

from imgaug import augmenters as iaa
import argparse
import os
import cv2
import shutil


def do_augmentation(images_path, output_path, num_times):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    for filename in os.listdir(images_path):
        images = []
        im = cv2.imread(os.path.sep.join([images_path, filename]))
        images.append(im)
        for i in range(1, num_times):
            color_seq = iaa.OneOf([
                iaa.HistogramEqualization(),
                #iaa.CLAHE()
                iaa.GaussianBlur(sigma=(0, 2.0))
            ])
            geo_seq = iaa.OneOf([
                #iaa.Flipud(0.5),
                iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-22.5, 22.5),
                    order=[0, 1],
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}
                    )
            ])

            color_aug_images = color_seq(images=images)
            geo_aug_images = geo_seq(images=images)

            output_name = os.path.splitext(filename)[0] + '_c' + str(i) + '.png'
            cv2.imwrite( os.path.sep.join([output_path, output_name]), color_aug_images[0] )

            output_name = os.path.splitext(filename)[0] + '_g' + str(i) + '.png'
            cv2.imwrite( os.path.sep.join([output_path, output_name]), geo_aug_images[0] )

        cv2.imwrite( os.path.sep.join([output_path, filename]), im )

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Script for data augmentation and class balance')
    argparser.add_argument('-i', '--images_path', help='path to images')
    argparser.add_argument('-o', '--output_path', help='output path')
    argparser.add_argument('-n', '--num_times', help='number of times the dataset will be expanded', type=int)
    args = argparser.parse_args()
    images_path = args.images_path
    output_path = args.output_path
    num_times = args.num_times
    do_augmentation(images_path, output_path, num_times)