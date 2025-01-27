from data_aug import do_augmentation

num_times = 2
class_list = ['0','1','2']
input_path = 'C:/Users/vntrolp/breast_density_classification/mias/train_path'
output_path = 'C:/Users/vntrolp/breast_density_classification/mias/train_path_aug'
for cl in class_list:
    images_input_path = input_path + '/' + cl
    images_output_path = output_path + '/' + cl
    do_augmentation(images_input_path, images_output_path, num_times)

input_path = 'C:/Users/vntrolp/breast_density_classification/mias/valid_path'
output_path = 'C:/Users/vntrolp/breast_density_classification/mias/valid_path_aug'
for cl in class_list:
    images_input_path = input_path + '/' + cl
    images_output_path = output_path + '/' + cl
    do_augmentation(images_input_path, images_output_path, num_times)