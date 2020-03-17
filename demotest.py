#!/usr/bin/env python
#  -*- coding: utf-8 -*-
import tensorflow as tf
import os, sys, cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg,read_cfg
from lib.fast_rcnn.test import im_detect
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer
from Arrdefine import *

this_dir = os.path.dirname(__file__)
path = os.path.join(this_dir,'config.txt')

CLASSES=read_cfg(path)

def endwith(s,*endstring):
   resultArray = map(s.endswith,endstring)
   if True in resultArray:
       return True
   else:
       return False

def vis_detections(im, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    equArr = []  # 单类设备数组集合
    for i in inds:
        imgRect = Img_Rectangle()  # 候选框坐标信息
        reco_equ = Reco_Equipment()  # 设备信息集合：候选框信息+设备类别（名称）
        bbox = dets[i, :4]
        score = dets[i, -1]
        imgRect.xmin = int(bbox[0])
        imgRect.ymin = int(bbox[1])
        imgRect.xmax = int(bbox[2])
        imgRect.ymax = int(bbox[3])
        imgRect.width = int(bbox[2]) - int(bbox[0])
        imgRect.height = int(bbox[3]) - int(bbox[1])
        reco_equ.equiName = class_name.upper()
        reco_equ.area = imgRect
        reco_equ.acreage = imgRect.width * imgRect.height
        reco_equ.score = score
        reco_equ.state = '1'
        # if endwith(reco_equ.equiName,'_ERROR'):
        #     reco_equ.state='0'
        #     reco_equ.equiName=reco_equ.equiName.replace('_ERROR','')
        # else:
        #     reco_equ.state = '1'


        equArr.append(reco_equ)

    return equArr


def demo(sess, net, image_name):
# def parse_args():
#     """Parse input arguments."""
#     parser = argparse.ArgumentParser(description='Faster R-CNN demo')
#     parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
#                         default=0, type=int)
#     parser.add_argument('--cpu', dest='cpu_mode',
#                         help='Use CPU mode (overrides --gpu)',
#                         action='store_true')
#     parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
#                         default='VGGnet_test')
#     # parser.add_argument('--model', dest='model', help='Model path',
#     #                     default='model/VGGnet_fast_rcnn_iter_60000.ckpt')
#
#     args = parser.parse_args()
#
#     return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    # args = parse_args()

    # if args.model == ' ' or not os.path.exists(args.model):
    #     print ('current path is ' + os.path.abspath(__file__))
    #     raise IOError(('Error: Model not found.\n'))

    # model = os.path.join(this_dir,'output','default','voc_2007_trainval','VGGnet_fast_rcnn_iter_100000.ckpt')

    model= os.path.join(this_dir,'model-6.16.ckpt')

    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network('VGGnet_test')
    # load model
    # print ('Loading network {:s}... '.format(args.demo_net)),
    saver = tf.train.Saver()
    saver.restore(sess, model)
    # print (' done.')
    # model = os.path.join(this_dir, 'model.ckpt')
    # saver.restore(sess, model)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = im_detect(sess, net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo42', '*.*'))

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {:s}'.format(im_name)
        demo(sess, net, im_name)

    plt.show()
