#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 17:27:38 2018

@author: caiom
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os.path
import bbox_utils
import cv2

eps = 0.00000001

classes_to_eval = {1 : "cow", 2: "diningtable", 3: "person", 4: "bottle", 5 : "chair", 6: "bus", 7 : "bird",
                 8 : "tvmonitor", 9 : "bicycle", 10 : "dog", 11 : "sofa", 12 : "sheep",13 : "horse", 14: "pottedplant",
                 15 : "cat", 16 : "train", 17 : "car", 18 : "boat", 19: "motorbike", 20 : "aeroplane"}


def ConvertJsonToImageBasedKey(jsonDict):
    """
    Make a dictionary where the key is the image
    """
    detections = {}

    assert 'image_id' in jsonDict[0], "Field 'image_id' not found in json file."

    for det in jsonDict:
        if det['image_id'] in detections:
            detections[det['image_id']].append(det)
        else:
            detections[det['image_id']] = [det]

    return detections


def CheckImageExtension(imgPath):
    if os.path.isfile(imgPath + '.jpg'):
        return '.jpg'
    if os.path.isfile(imgPath + '.jpeg'):
        return '.jpeg'
    if os.path.isfile(imgPath + '.png'):
        return '.png'

    assert False, 'Image file extension not found: ' + imgPath


def evaluate_numpy_img(img_name, img_np, gt_boxes_dict, pred_boxes_dict, total_detection_scores=None,
                       total_class_scores=None, plot_img=False, verbose=False, resizeFactor=1.0):
    if plot_img:
        fig, ax = plt.subplots(1)
        ax.imshow(img_np)

    n_dont_care = 0

    class_scores = {key: {'n_gt': 0, 'n_pred': 0, 'correct': 0} for key in classes_to_eval}

    # Get the number of dont care boxes so we can create matrices with the right size
    for item in gt_boxes_dict:
        if item['category_id'] == 23:
            n_dont_care += 1

    # The number 4 is for the bound box. The size 5 is for easy of plotting
    gp_boxes_dont_care_np = np.empty((n_dont_care, 4))
    gt_boxes_np = np.empty((len(gt_boxes_dict) - n_dont_care, 5))
    pred_boxes_all_np = np.empty((len(pred_boxes_dict), 5))

    # Place the boxes in the correct format for bbox_overlaps
    n_item_valid = 0
    n_item_dont_care = 0
    for n_item, item in enumerate(gt_boxes_dict):
        bboxd = item['bbox']

        for cordN, cord in enumerate(bboxd):
            bboxd[cordN] = cord * resizeFactor

        if item['category_id'] != 23:
            if bboxd[2] * bboxd[3] < 1:
                print("Error on bbox size, bbox %d" % (n_item))
            gt_boxes_np[n_item_valid, 0] = bboxd[0]
            gt_boxes_np[n_item_valid, 1] = bboxd[1]
            gt_boxes_np[n_item_valid, 2] = bboxd[0] + bboxd[2]
            gt_boxes_np[n_item_valid, 3] = bboxd[1] + bboxd[3]
            gt_boxes_np[n_item_valid, 4] = item['category_id']
            n_item_valid += 1
        else:
            gp_boxes_dont_care_np[n_item_dont_care, 0] = bboxd[0]
            gp_boxes_dont_care_np[n_item_dont_care, 1] = bboxd[1]
            gp_boxes_dont_care_np[n_item_dont_care, 2] = bboxd[0] + bboxd[2]
            gp_boxes_dont_care_np[n_item_dont_care, 3] = bboxd[1] + bboxd[3]
            n_item_dont_care += 1
            if plot_img:
                rect = patches.Rectangle((bboxd[0], bboxd[1]), bboxd[2], bboxd[3], linewidth=1, edgecolor='yellow',
                                         facecolor='none')
                ax.add_patch(rect)

    for n_item, item in enumerate(pred_boxes_dict):
        bboxd = item['bbox']
        pred_boxes_all_np[n_item, 0] = bboxd[0]
        pred_boxes_all_np[n_item, 1] = bboxd[1]
        pred_boxes_all_np[n_item, 2] = bboxd[0] + bboxd[2]
        pred_boxes_all_np[n_item, 3] = bboxd[1] + bboxd[3]
        pred_boxes_all_np[n_item, 4] = item['category_id']
        if plot_img:
            rect = patches.Rectangle((bboxd[0], bboxd[1]), bboxd[2], bboxd[3], linewidth=1, edgecolor='purple',
                                     facecolor='none')
            ax.add_patch(rect)

    # First let consider the dont car class and discard all predictions that have an overlap
    iou = bbox_utils.bbox_overlaps(gp_boxes_dont_care_np, pred_boxes_all_np)

    if iou.shape[0] > 0:
        dont_care_preds_arg_max = np.max(iou, axis=0)
        dont_care_preds_index = np.where(dont_care_preds_arg_max < eps)
        pred_boxes_np = pred_boxes_all_np[dont_care_preds_index, :][0]
    else:
        pred_boxes_np = pred_boxes_all_np

    # Overlap between boxes
    iou = bbox_utils.bbox_overlaps(gt_boxes_np, pred_boxes_np)

    valid_matches = 0

    if pred_boxes_np.shape[0] == 0:
        print('All boxes are dont care')
        return

    # For each gt box, what is the prediction box with the maximum iou
    maxes_iou_index_gt = np.argmax(iou, axis=1)
    # For each prediction box, what is the gt with the maximum iou
    maxes_iou_index_pred = np.argmax(iou, axis=0)

    # A helper to calculate matches
    range_iou_gt = np.arange(maxes_iou_index_gt.shape[0])

    # Here we check if the maximum for the gt perspective and from the prediction perspective match
    # index_match_on_pred contains valid (gt indices) of those matches
    index_match_gt = np.where(maxes_iou_index_pred[maxes_iou_index_gt] == range_iou_gt)[0]

    # Get the iou of the matches
    matches_iou = iou[index_match_gt, maxes_iou_index_gt[index_match_gt]]

    # Threshold
    valid_matches = matches_iou > 0.5

    # Get matched preds and gts indices
    true_preds_index = maxes_iou_index_gt[index_match_gt][valid_matches]
    true_gts_index = index_match_gt[valid_matches]

    # Create a mask (0, 1) for the predictions
    mask_preds = np.zeros(iou.shape[1], dtype=bool)
    mask_preds[true_preds_index] = True

    mask_gt = np.zeros(iou.shape[0], dtype=bool)
    mask_gt[true_gts_index] = True

    # Get the actual values and plot
    true_preds_np = pred_boxes_np[true_preds_index]
    false_preds_np = pred_boxes_np[~mask_preds]

    true_gt_np = gt_boxes_np[true_gts_index]
    false_gt_np = gt_boxes_np[~mask_gt]

    if plot_img:

        for i in range(true_preds_np.shape[0]):
            rect = patches.Rectangle((true_preds_np[i, 0], true_preds_np[i, 1]),
                                     true_preds_np[i, 2] - true_preds_np[i, 0],
                                     true_preds_np[i, 3] - true_preds_np[i, 1], linewidth=1, edgecolor='green',
                                     facecolor='none')
            ax.add_patch(rect)

        for i in range(false_preds_np.shape[0]):
            rect = patches.Rectangle((false_preds_np[i, 0], false_preds_np[i, 1]),
                                     false_preds_np[i, 2] - false_preds_np[i, 0],
                                     false_preds_np[i, 3] - false_preds_np[i, 1], linewidth=1, edgecolor='white',
                                     facecolor='none')
            ax.add_patch(rect)

        for i in range(false_gt_np.shape[0]):
            rect = patches.Rectangle((false_gt_np[i, 0], false_gt_np[i, 1]), false_gt_np[i, 2] - false_gt_np[i, 0],
                                     false_gt_np[i, 3] - false_gt_np[i, 1], linewidth=1, edgecolor='red',
                                     facecolor='none')
            ax.add_patch(rect)

        for i in range(true_gt_np.shape[0]):
            rect = patches.Rectangle((true_gt_np[i, 0], true_gt_np[i, 1]), true_gt_np[i, 2] - true_gt_np[i, 0],
                                     true_gt_np[i, 3] - true_gt_np[i, 1], linewidth=1, edgecolor='blue',
                                     facecolor='none')
            ax.add_patch(rect)

    num_true_positive = float(np.sum(valid_matches))

    if total_detection_scores:
        total_detection_scores['correct'] += num_true_positive
        total_detection_scores['n_gt'] += iou.shape[0]
        total_detection_scores['n_pred'] += iou.shape[1]

    recall = num_true_positive / iou.shape[0]
    precision = num_true_positive / (iou.shape[1] + eps)
    fscore = 2 * ((precision * recall) / (precision + recall + eps))

    true_classes = true_gt_np[..., 4].astype(np.int64)
    pred_classes = true_preds_np[..., 4].astype(np.int64)

    for i in range(true_classes.size):
        if true_classes[i] in class_scores and pred_classes[i] in class_scores:
            class_scores[true_classes[i]]['n_gt'] += 1
            class_scores[pred_classes[i]]['n_pred'] += 1
            if true_classes[i] == pred_classes[i]:
                class_scores[pred_classes[i]]['correct'] += 1
                if plot_img:
                    ax.text(true_gt_np[i, 0], true_gt_np[i, 1], 'T: %s' % classes_to_eval[true_classes[i]], fontsize=5,
                            weight='bold', color='green')
            else:
                if plot_img:
                    ax.text(true_gt_np[i, 0], true_gt_np[i, 1],
                            'F: (%s/%s)' % (classes_to_eval[true_classes[i]], classes_to_eval[pred_classes[i]]),
                            fontsize=5, weight='bold', color='red')
    #        else:
    #            assert False, 'A non class is present (%d or %d).' % (true_classes[i], pred_classes[i])

    if plot_img:
        plt.suptitle(img_name, fontsize=16)
        plt.show()
    class_accuracy = np.sum(np.equal(true_classes, pred_classes)) / true_preds_np.shape[0]

    if total_class_scores:
        for key in total_class_scores:
            total_class_scores[key]['n_gt'] += class_scores[key]['n_gt']
            total_class_scores[key]['correct'] += class_scores[key]['correct']
            total_class_scores[key]['n_pred'] += class_scores[key]['n_pred']

    if verbose:
        print('Detection Precision %d/%d - %.3f' % (num_true_positive, iou.shape[1], precision))
        print('Detection Recall %d/%d - %.3f' % (num_true_positive, iou.shape[0], recall))
        print('Detection F-score %.3f' % (fscore))
        print('Class accuracy: %.3f' % (class_accuracy))
        avg_class_fscore = 0.

        for key in class_scores:
            print('Eval for class %s:' % (classes_to_eval[key]))
            precision = class_scores[key]['correct'] / (class_scores[key]['n_pred'] + eps)
            recall = class_scores[key]['correct'] / (class_scores[key]['n_gt'] + eps)
            fscore = 2 * ((precision * recall) / (precision + recall + eps))
            avg_class_fscore += fscore
            print('Class %s Precision %.3f (%d/%d)' % (
            classes_to_eval[key], precision, class_scores[key]['correct'], class_scores[key]['n_pred']))
            print('Class %s Recall %.3f (%d/%d)' % (
            classes_to_eval[key], recall, class_scores[key]['correct'], class_scores[key]['n_gt']))
            print('Class %s F-score %.3f' % (classes_to_eval[key], fscore))

        print('Average class F-score %.3f.' % (avg_class_fscore / len(total_class_scores)))


def EvaluateDetectionsOnTestSet(predJsonPath, gtJsonPath, imgsPath, plotImgs=False, verbose=False, resizeFactor=1.0):
    """
    Given the results of a detection in json format and the gt in json format evaluate the results.

    Args:
        predJsonPath (str): Path to the predictions json file.
        gtJsonPath (str): Path to the gt file.
        imgsPath (str): Path to the images files that the gt references. Needed to plot.
        plotImgs (bool): Plot images ?
        verbose (bool): print scores and additional information.
        resizeFactor (float): img resize factor of the predictions with repect to the gt.
    Returns:
        fscore (float): The total fscore
    """

    if type(predJsonPath) is str:
        detectections_json = json.load(open(predJsonPath, 'r'))
    else:
        detectections_json = predJsonPath

    detections = ConvertJsonToImageBasedKey(detectections_json)

    ground_truth_json = json.load(open(gtJsonPath, 'r'))
    ground_truth_json = ground_truth_json['annotations']
    ground_truth = ConvertJsonToImageBasedKey(ground_truth_json)

    total_detection_scores = {'n_gt': 0, 'n_pred': 0, 'correct': 0}
    total_class_scores = {key: {'n_gt': 0, 'n_pred': 0, 'correct': 0} for key in classes_to_eval}

    for image in ground_truth:

        if verbose:
            print('Evaluating image with id: %s.' % (image))

        gt_boxes_dict = ground_truth[image]
        if image in detections:
            pred_boxes_dict = detections[image]
        else:
            pred_boxes_dict = []
        img_np = None

        if plotImgs:
            ext = CheckImageExtension(os.path.join(imgsPath, image))
            img_np = cv2.imread(os.path.join(imgsPath, image + ext), cv2.IMREAD_IGNORE_ORIENTATION + cv2.IMREAD_COLOR)
            img_np = img_np[:, :, ::-1]

        evaluate_numpy_img(image, img_np, gt_boxes_dict, pred_boxes_dict, total_detection_scores, total_class_scores,
                           plotImgs, verbose)

    precision = total_detection_scores['correct'] / total_detection_scores['n_pred']
    recall = total_detection_scores['correct'] / total_detection_scores['n_gt']
    fscore_detection = 2 * ((precision * recall) / (precision + recall + eps))

    final_str = ''

    final_str += '\n-------- TOTAL --------\n\n'
    final_str += 'Total Detection Precision %d/%d - %.3f\n' % (
    total_detection_scores['correct'], total_detection_scores['n_pred'], precision)
    final_str += 'Total Detection Recall %d/%d - %.3f\n' % (
    total_detection_scores['correct'], total_detection_scores['n_gt'], recall)
    final_str += 'Total Detection F-score %.3f\n' % (fscore_detection)

    avg_class_fscore = 0.
    for key in total_class_scores:
        precision = total_class_scores[key]['correct'] / (total_class_scores[key]['n_pred'] + eps)
        recall = total_class_scores[key]['correct'] / (total_class_scores[key]['n_gt'] + eps)
        fscore_class = 2 * ((precision * recall) / (precision + recall + eps))
        avg_class_fscore += fscore_class

        final_str += 'Total Eval for class %s:\n' % (classes_to_eval[key])
        final_str += 'Total Class %s Precision %.3f (%d/%d)\n' % (
        classes_to_eval[key], precision, total_class_scores[key]['correct'], total_class_scores[key]['n_pred'])
        final_str += 'Total Class %s Recall %.3f (%d/%d)\n' % (
        classes_to_eval[key], recall, total_class_scores[key]['correct'], total_class_scores[key]['n_gt'])
        final_str += 'Total Class %s F-score %.3f\n' % (classes_to_eval[key], fscore_class)

    avg_class_fscore /= len(classes_to_eval)
    final_str += 'Total average class F-score %.3f.\n' % (avg_class_fscore)

    if verbose:
        print(final_str)

    return fscore_detection, avg_class_fscore, final_str


if __name__ == "__main__":
    import os, inspect

    repository_folder = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))

    resizeFactor = 1.0
    imgsPath = os.path.join(repository_folder, 'RCNN/EvaluationImages')
    predJsonPath = os.path.join('/home/vitor/python/Detectron.pytorch/tools/detections_100.json')
    gtJsonPath = os.path.join('/home/vitor/python/Detectron.pytorch/data/VOC2007_test/VOCdevkit2007/VOC2007/Annotations/VOC2007_test_2.json')
    plotImgs = True
    verbose = True

    EvaluateDetectionsOnTestSet(predJsonPath, gtJsonPath, imgsPath, plotImgs, verbose, resizeFactor)







