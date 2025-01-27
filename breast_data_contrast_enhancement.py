from contrast_enhancement import do_contrast_enhancement

class_list = ['0','1','2']
input_path = 'C:/Users/vntrolp/breast_density_classification/mias/train_path'
output_path = 'C:/Users/vntrolp/breast_density_classification/mias/train_path_enh'
for cl in class_list:
    images_input_path = input_path + '/' + cl
    images_output_path = output_path + '/' + cl
    do_contrast_enhancement(images_input_path, images_output_path)

input_path = 'C:/Users/vntrolp/breast_density_classification/mias/valid_path'
output_path = 'C:/Users/vntrolp/breast_density_classification/mias/valid_path_enh'
for cl in class_list:
    images_input_path = input_path + '/' + cl
    images_output_path = output_path + '/' + cl
    do_contrast_enhancement(images_input_path, images_output_path)

input_path = 'C:/Users/vntrolp/breast_density_classification/mias/test_path'
output_path = 'C:/Users/vntrolp/breast_density_classification/mias/test_path_enh'
for cl in class_list:
    images_input_path = input_path + '/' + cl
    images_output_path = output_path + '/' + cl
    do_contrast_enhancement(images_input_path, images_output_path)