#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# Edited by Matthew Seals
# --------------------------------------------------------
"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect

from torchvision.ops import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import argparse
from matplotlib import cm

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

import torch

CLASSES = ('__background__', 'person', 'cellphone', 'computer', 'radar', 'palmtree')

NETS = {
    'vgg16': ('vgg16_faster_rcnn_iter_%d.pth', ),
    'res101': ('res101_faster_rcnn_iter_%d.pth', ),
    'res50': ('res50_faster_rcnn_iter_%d.pth', )
}
DATASETS = {
    'pascal_voc': ('voc_2007_trainval', ),
    'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval', ),
    'aic': ('aic_trainval', )
}

COLORS = [cm.tab10(i) for i in np.linspace(0., 1., 10)]


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(
        timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    thresh = 0.8  # CONF_THRESH
    NMS_THRESH = 0.3

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    cntr = -1

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(
            torch.from_numpy(cls_boxes), torch.from_numpy(cls_scores),
            NMS_THRESH)
        dets = dets[keep.numpy(), :]
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            continue
        else:
            cntr += 1

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1],
                              fill=False,
                              edgecolor=COLORS[cntr % len(COLORS)],
                              linewidth=3.5))
            ax.text(
                bbox[0],
                bbox[1] - 2,
                '{:s} {:.3f}'.format(cls, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14,
                color='white')

        ax.set_title(
            'All detections with threshold >= {:.1f}'.format(thresh),
            fontsize=14)

        plt.axis('off')
        plt.tight_layout()
    save_path = os.path.join(os.getcwd(), 'data/demo_det/', 'demo_' + image_name)
    plt.savefig(save_path)
    print('Saved to `{}`'.format(save_path))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Tensorflow Faster R-CNN demo')
    parser.add_argument(
        '--net',
        dest='demo_net',
        help='Network to use [vgg16 res101 res50]',
        choices=NETS.keys(),
        default='res101')
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='Trained dataset [pascal_voc pascal_voc_0712 aic]',
        choices=DATASETS.keys(),
        default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    saved_model = os.path.join(
        'output', demonet, DATASETS[dataset][0], 'default',
        NETS[demonet][0] % (70000 if dataset == 'pascal_voc' or 'aic' else 110000))
        # NETS[demonet][0] % (49000 if dataset == 'aic' else 110000))

    if not os.path.isfile(saved_model):
        raise IOError(
            ('{:s} not found.\nDid you download the proper networks from '
             'our server and place them properly?').format(saved_model))

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    elif demonet == 'res50':
        net = resnetv1(num_layers=50)
    else:
        raise NotImplementedError
    net.create_architecture(len(CLASSES), tag='default', anchor_scales=[8, 16, 32])

    net.load_state_dict(torch.load(saved_model))

    net.eval()
    net.cuda()

    print('Loaded network {:s}'.format(saved_model))

    demo_dir = os.path.join(cfg.DATA_DIR, 'demo')
    im_names = os.listdir(demo_dir)

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(net, im_name)

    plt.show()
