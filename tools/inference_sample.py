from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os.path as osp
import sys, inspect

import argparse
import distutils.util
import os
import sys
import pprint
import subprocess
from collections import defaultdict
from six.moves import xrange


import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable

import _init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer
import matplotlib.pyplot as plt
import json, time, math, os


# Add lib path

class_to_name = {1 : "cow", 2: "diningtable", 3: "person", 4: "bottle", 5 : "chair", 6: "bus", 7 : "bird",
                 8 : "tvmonitor", 9 : "bicycle", 10 : "dog", 11 : "sofa", 12 : "sheep",13 : "horse", 14: "pottedplant",
                 15 : "cat", 16 : "train", 17 : "car", 18 : "boat", 19: "motorbike", 20 : "aeroplane"}


current_folder = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
repository_folder = os.path.dirname(current_folder)

image_folder = os.path.join("/home/vitor/python/Detectron.pytorch/sample_detection")
config_file = os.path.join("/home/vitor/python/Detectron.pytorch/configs/pcs5886/e2e_faster_rcnn_R-50-FPN_1x.yaml")
weights = os.path.join("/home/vitor/python/Detectron.pytorch/Outputs/e2e_faster_rcnn_R-50-FPN_1x/Dec01-22-20-25_vitor-Z370M-AORUS-Gaming_step/ckpt/model_step17192.pth")
output_folder = os.path.join("/home/vitor/faster-rcnn-outputs/10000steps")
json_file_path = os.path.join(current_folder, 'detections_100.json')

json_data = []
# Create output folder if it does not exist
if not osp.exists(output_folder):
    os.mkdir(output_folder)

if not torch.cuda.is_available():
    sys.exit("Need a CUDA device to run the code.")

cfg.MODEL.NUM_CLASSES = 21

cfg_from_file(config_file)
assert_and_infer_cfg()

maskRCNN = Generalized_RCNN().cuda()

load_name = weights
print("loading checkpoint %s" % load_name)
checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
net_utils.load_ckpt(maskRCNN, checkpoint['model'])

maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                             minibatch=True, device_ids=[0])  # only support single GPU

maskRCNN.eval()

img_list = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and (
            f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg"))]

for im_name in img_list:

    t = time.time()
    im = cv2.imread(osp.join(image_folder, im_name), cv2.IMREAD_IGNORE_ORIENTATION + cv2.IMREAD_COLOR)
    out_img = im.copy()

    cls_boxes, cls_segms, cls_keyps = im_detect_all(maskRCNN, im)

    t = time.time()
    threshold = 0.5
    n_det = 0

    detection_list = []

    for n_class in range(1, cfg.MODEL.NUM_CLASSES):
        for temp_box in cls_boxes[n_class]:
            if temp_box[4] > threshold:
                gt_class = n_class
                detection = {'xMax': max(temp_box[0], temp_box[2]), 'xMin': min(temp_box[0], temp_box[2]),
                             'yMax': max(temp_box[1], temp_box[3]), 'yMin': min(temp_box[1], temp_box[3]),
                             'class': class_to_name[gt_class],
                             'area': math.fabs((temp_box[0] - temp_box[2]) * (temp_box[1] - temp_box[3])),
                             'perimeter': (2 * math.fabs(temp_box[0] - temp_box[2])
                                           + 2 * math.fabs(temp_box[1] - temp_box[3])),
                             'tileName': osp.basename(im_name),
                             'score': int(temp_box[4]),
                             'class_id': gt_class}
                detection_list.append(detection)
                n_det += 1

    spend_time = time.time() - t
    print('Time to get Bounding Boxes: %.3f' % spend_time)
    print('Number of Objects: %d.' % n_det)
    print('Image: %s.' % im_name)

    pred_boxes_np = np.empty((len(detection_list), 4))

    for n_bbox, bbox in enumerate(detection_list):
        pred_boxes_np[n_bbox, 0] = bbox['xMin']
        pred_boxes_np[n_bbox, 1] = bbox['yMin']
        pred_boxes_np[n_bbox, 2] = bbox['xMax']
        pred_boxes_np[n_bbox, 3] = bbox['yMax']


    for bbox in detection_list:
        cv2.rectangle(im, (int(bbox['xMin'] - 5), int(bbox['yMin'] - 5)),
                      (int(bbox['xMax'] + 5), int(bbox['yMax'] + 5)), (255, 0, 0), 5)

        category = bbox['class_id']

        bbox_height = int(bbox['yMax']) - int(bbox['yMin'])
        bbox_width = int(bbox['xMax']) - int(bbox['xMin'])
        json_data.append({'image_id': int('2018' + str(im_name.split('.')[-2]).zfill(7)) , 'category_id': category,
                          'bbox': [int(bbox['xMin']), int(bbox['yMin']), bbox_width, bbox_height]})

    #cv2.putText(out_img, im_name, (230, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 4, cv2.LINE_AA)
    plt.figure()
    plt.imshow(out_img[..., ::-1])
    plt.show()

    cv2.imwrite(osp.join(output_folder, im_name), out_img)

    print('Inference time: %.3f' % (time.time() - t))

with open(json_file_path, 'w') as fp:
    json.dump(json_data, fp)

