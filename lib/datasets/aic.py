# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import subprocess
import uuid
from .aic_eval import aic_eval
from model.config import cfg


class aic(imdb):
    def __init__(self, image_set, use_diff=False):
        name = 'aic_' + image_set
        if use_diff:
            name += '_diff'
        imdb.__init__(self, name)
        self._image_set = image_set
        self._devkit_path = self._get_default_path()
        self._data_path = os.path.join(self._devkit_path, 'data')
        self._classes = (
            '__background__',  # always index 0
            'person',
            'cellphone',
            'computer',
            'radar',
            'palmtree')
        self._class_to_ind = dict(
            list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.JPEG'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {
            'cleanup': True,
            'use_salt': True,
            'use_diff': use_diff,
            'rpn_file': None
        }

        assert os.path.exists(self._devkit_path), \
          'AIC_devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
          'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
    Return the absolute path to image i in the image sequence.
    """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
    Construct an image path from the image's "index" identifier.
    """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
          'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
    Load the indexes listed in this dataset's image set file.
    """
        # Example path to image set file:
        # self._data_path + /ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
          'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
    Return the default path where PASCAL VOC is expected to be installed.
    """
        return os.path.join(cfg.DATA_DIR, 'AIC_devkit')

    def gt_roidb(self):
        """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [
            self._load_aic_annotation(index) for index in self.image_index
        ]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
          'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_aic_annotation(self, index):
        """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        tree_objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            # non_diff_objs = [obj for obj in tree_objs if int(obj.find('difficult').text) == 0]
            # # if len(non_diff_objs) != len(objs):
            # #     print 'Removed {} difficult objects'.format(
            # #         len(objs) - len(non_diff_objs))
            # objs = non_diff_objs

            objs = []
            for obj in tree_objs:
                if obj.find('difficult'):
                    if int(obj.find('difficult').text) == 0:
                        objs.append(obj)
                else:
                    objs.append(obj)

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            # x1 = float(bbox.find('xmin').text) - 1
            # y1 = float(bbox.find('ymin').text) - 1
            # x2 = float(bbox.find('xmax').text) - 1
            # y2 = float(bbox.find('ymax').text) - 1
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)

            cls = self._class_to_ind[obj.find('name').text.strip()]

            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas
        }

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt
                   if self.config['use_salt'] else self._comp_id)
        return comp_id

    def _get_aic_results_file_template(self):
        # AIC_devkit/results/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id(
        ) + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(self._devkit_path, 'results', filename)
        return path

    def _write_aic_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} AIC results file'.format(cls))
            filename = self._get_aic_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    # for k in range(dets.shape[0]):
                    #     f.write(
                    #         '{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                    #             index, dets[k, -1], dets[k, 0] + 1,
                    #             dets[k, 1] + 1, dets[k, 2] + 1,
                    #             dets[k, 3] + 1))
                    for k in range(dets.shape[0]):
                        f.write(
                            '{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                                index, dets[k, -1], dets[k, 0],
                                dets[k, 1], dets[k, 2],
                                dets[k, 3]))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(self._devkit_path, 'data', 
                                    'Annotations', '{:s}.xml')
        imagesetfile = os.path.join(self._devkit_path, 'data',
                                    'ImageSets', 'Main',
                                    self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_aic_results_file_template().format(cls)
            rec, prec, ap = aic_eval(
                filename,
                annopath,
                imagesetfile,
                cls,
                cachedir,
                ovthresh=0.5,
                use_diff=self.config['use_diff'])
            aps += [ap]
            print(('AP for {} = {:.4f}'.format(cls, ap)))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print(('Mean AP = {:.4f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print(('{:.3f}'.format(ap)))
        print(('{:.3f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_aic_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_aic_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    from datasets.aic import aic

    d = aic('trainval')
    res = d.roidb
    from IPython import embed

    embed()
