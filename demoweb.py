#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os, cv2
import argparse
import os.path as osp
import base64

this_dir = osp.dirname(__file__)
print(this_dir)

from lib.fast_rcnn.config import read_cfg
from lib.fast_rcnn.test import im_detect
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer
from Arrdefine import *


this_dir = os.path.dirname(__file__)
path = os.path.join(this_dir,'config.txt')

CLASSES=read_cfg(path)

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

def endwith(s,*endstring):
   resultArray = map(s.endswith,endstring)
   if True in resultArray:
       return True
   else:
       return False

def getRes_Img(sess, net, image):
    """Detect object classes in an image using pre-computed object proposals."""

    imgCon=image.imgcontent
    imgString = base64.b64decode(imgCon)
    nparr = np.fromstring(imgString,np.uint8)
    im = cv2.imdecode(nparr,cv2.IMREAD_COLOR)

    # im=cv2.imread(image)
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    im = im[:, :, (2, 1, 0)]

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    res_img = Res_Image()    #初始化，图片中所有设备信息集合：设备信息+图片名称（编号）

    equiAllArr = []    #图片中所有设备的数组集合
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        equiarr=EquiArr(im, cls, dets, thresh=CONF_THRESH)    #只能给出某一类设备的所有候选框的集合

        if equiarr != None:
            for x in equiarr:
                equiAllArr.append(x)    #将一张图片中所有设备的信息整合到一个数组中

    equiAllArr.sort(key=lambda Reco_Equipment: Reco_Equipment.acreage,reverse = False)
    equiAllArrNew = []    #嵌套数组
    ds=[]         #数组索引

    #候选框是并列的,人为的创建嵌套数组,
    for i in range(len(equiAllArr)):
        if '_' not in equiAllArr[i].equiName:
            if (i in ds):  # 已经访问过的不再访问，直接跳过进行下一个
                continue
            ds.append(i)
            xmin = equiAllArr[i].area.xmin
            ymin = equiAllArr[i].area.ymin
            xmax = equiAllArr[i].area.xmax
            ymax = equiAllArr[i].area.ymax
            equChilds = []  # 子集，即被嵌套的设备集合
            for m in range(0, len(equiAllArr)):
                if (m in ds):
                    continue
                xx1 = np.maximum(xmin, equiAllArr[m].area.xmin)
                yy1 = np.maximum(ymin, equiAllArr[m].area.ymin)
                xx2 = np.minimum(xmax, equiAllArr[m].area.xmax)
                yy2 = np.minimum(ymax, equiAllArr[m].area.ymax)
                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)
                inter = float(w * h)  # 重叠部分面积
                if (inter / equiAllArr[m].acreage >= 0.8)and equiAllArr[i].equiName=='DLQ' and endwith(equiAllArr[m].equiName,'_CT'):
                    h1=np.maximum(ymin, equiAllArr[m].area.ymin)-np.minimum(ymin, equiAllArr[m].area.ymin)
                    h2= np.maximum(ymax, equiAllArr[m].area.ymax)- np.minimum(ymax, equiAllArr[m].area.ymax)
                    if h1<=h2:
                        child = Reco_Equipment_child()
                        child.equiName = equiAllArr[i].equiName + '_MHCT'
                        child.area = equiAllArr[m].area
                        equChilds.append(child)
                        ds.append(m)
                    else:
                        child = Reco_Equipment_child()
                        child.equiName = equiAllArr[i].equiName + '_ZZCT'
                        child.area = equiAllArr[m].area
                        equChilds.append(child)
                        ds.append(m)
                elif (inter / equiAllArr[m].acreage >= 0.8) and endwith(equiAllArr[m].equiName, '_CT'):
                        child = Reco_Equipment_child()
                        child.equiName = equiAllArr[i].equiName + '_CT'
                        child.area = equiAllArr[m].area
                        equChilds.append(child)
                        ds.append(m)
                elif (inter / equiAllArr[m].acreage >= 0.8) and endwith(equiAllArr[m].equiName,'_JT'):
                    child = Reco_Equipment_child()
                    child.equiName = equiAllArr[i].equiName+'_JT'
                    child.area = equiAllArr[m].area
                    equChilds.append(child)
                    ds.append(m)
                elif (inter / equiAllArr[m].acreage >= 0.8):
                    child = Reco_Equipment_child()
                    child.equiName = equiAllArr[m].equiName
                    child.area = equiAllArr[m].area
                    equChilds.append(child)
                    ds.append(m)


            equiAllArr[i].children = equChilds
            equiAllArrNew.append(equiAllArr[i])

    for i in range(len(equiAllArr)):
        if (i in ds):  # 已经访问过的不再访问，直接跳过进行下一个
            continue
        equChilds = []
        equiAllArr[i].children = equChilds
        equiAllArrNew.append(equiAllArr[i])

    res_img.imgID = image.imgID
    # 按照条件选取需要的内容
    if image.equiptype == '':
        res_img.equipments = equiAllArrNew
    else:
        equiresArr = []
        res_equip = str.lower(image.equiptype)
        for n in range(len(equiAllArrNew)):
            if res_equip in equiAllArrNew[n].equiName:
                equiresArr.append(equiAllArrNew[n])
        res_img.equipments = equiresArr

    return res_img

def EquiArr(im,class_name, dets,thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    equArr=[]    #单类设备数组集合
    # pic=[]
    for i in inds:
        imgRect = Img_Rectangle()    #候选框坐标信息
        reco_equ = Reco_Equipment()   #设备信息集合：候选框信息+设备类别（名称）

        bbox = dets[i, :4]

        imgRect.xmin=int(bbox[0])
        imgRect.ymin = int(bbox[1])
        imgRect.xmax = int(bbox[2])
        imgRect.ymax = int(bbox[3])
        imgRect.width = int(bbox[2]) - int(bbox[0])
        imgRect.height = int(bbox[3]) - int(bbox[1])
        reco_equ.equiName=class_name.upper()
        reco_equ.area=imgRect
        reco_equ.acreage=imgRect.width*imgRect.height

        # picture = im[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        # picture = cv2.resize(picture,(128,128),interpolation=cv2.INTER_AREA)
        # picType = model.predict(picture)
        #设备图片必须是成对的(即同一设备正常和故障文件夹必须同时存在，哪怕文件夹里没有图片，这样CNN训练和测试时标签才能对应上)
        if endwith(reco_equ.equiName,'_ERROR'):
            reco_equ.state='0'
            reco_equ.equiName=reco_equ.equiName.replace('_ERROR','')
        else:
            reco_equ.state = '1'

        equArr.append(reco_equ)

    return equArr
