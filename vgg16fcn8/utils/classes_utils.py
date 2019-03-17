import numpy as np


def get_num_classes(hypes):
    return len(hypes['classes'])


def get_color_classes(hypes):
    color_classes = []
    for classes in hypes['classes']:
        color_classes.append(np.array(classes['colors']))
    return color_classes


def get_output_color_classes(hypes):
    color_classes = []
    for classes in hypes['classes']:
        color_classes.append(np.array(classes['output']))
    return color_classes


def get_output_color_dict(hypes):
    color_classes = get_output_color_classes(hypes)
    color_dict = {}
    for i in range(len(color_classes)):
        color_dict[i] = tuple(color_classes[i])
    return color_dict


def get_name_classes(hypes):
    name_classes = []
    for classes in hypes['classes']:
        name_classes.append(classes['name'])
    return name_classes


def get_gt_image_index(gt_image, hypes):
    color_classes = get_color_classes(hypes)
    result = np.ndarray(shape=gt_image.shape[:2], dtype=int)
    result[:, :] = 0
    i = 0
    for color_class in color_classes:
        for color in color_class:
            result[(gt_image == color).all(2)] = i
        i += 1
    return result
