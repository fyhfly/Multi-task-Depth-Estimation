import numpy as np
import torch
from torch.autograd import Variable
from math import ceil
from PIL import Image
import torch.nn as nn

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
ignore_label = 255
id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                  3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                  7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                  14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                  18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                  28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

def id2trainId(label, id_to_trainid, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img

def predict_sliding(net, image, tile_size, classes, flip_evaluation, recurrence):
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    image_size = image.shape
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    #print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = np.zeros((image_size[2], image_size[3], classes))
    count_predictions = np.zeros((image_size[2], image_size[3], classes))
    tile_counter = 0

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], image_size[3])
            y2 = min(y1 + tile_size[0], image_size[2])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = image[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            padded_prediction = net(Variable(torch.from_numpy(padded_img), volatile=True).cuda())
            if isinstance(padded_prediction, list):
                padded_prediction = padded_prediction[0]
            padded_prediction = interp(padded_prediction).cpu().data[0].numpy().transpose(1,2,0)
            prediction = padded_prediction[0:img.shape[2], 0:img.shape[3], :]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

    full_probs /= count_predictions
    return full_probs


def get_confusion_matrix(gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix

def evaluate_semantic(gt_semantic, pred_semantic):
    """start the evaluation process."""
    num_classes = 19

    confusion_matrix = np.zeros((num_classes,num_classes))
    palette = get_palette(256)
    output = id2trainId(pred_semantic, id_to_trainid, reverse=True)
    output = Image.fromarray(output.astype(np.uint8), 'L')
    output.putpalette(palette)

    ignore_index = gt_semantic != 255
    gt_semantic = gt_semantic[ignore_index]
    pred_semantic = pred_semantic[ignore_index]
    confusion_matrix += get_confusion_matrix(gt_semantic, pred_semantic, num_classes)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IU_array = (tp/np.maximum(1.0, pos+res-tp))
    mean_IU = IU_array.mean()


    return mean_IU, output, IU_array

# def evaluate_semantic(gt_semantic, pred_semantic):
#     num_classes = 19
#
#     gt_copy = gt_semantic.copy()
#     pred_copy = pred_semantic.copy()
#
#     gt_copy += 1
#     pred_copy += 1
#     pred_copy = pred_copy * (gt_copy > 0)
#
#     intersection = pred_copy * (pred_copy == gt_copy)
#     (area_intersection, _) = np.histogram(intersection, bins=num_classes, range=(1, num_classes))
#
#     (area_pred, _) = np.histogram(pred_copy, bins=num_classes, range=(1, num_classes))
#     (area_gt, _) = np.histogram(gt_copy, bins=num_classes, range=(1, num_classes))
#     area_union = area_pred + area_gt - area_intersection
#
#     IU = area_intersection.sum() / area_union.sum()
#
#     palette = get_palette(256)
#     pred_semantic = id2trainId(pred_semantic, id_to_trainid, reverse=True)
#     output = Image.fromarray(pred_semantic.astype(np.uint8), 'L')
#     output.putpalette(palette)
#
#     return IU, output
