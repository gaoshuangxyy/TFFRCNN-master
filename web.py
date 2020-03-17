#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import tensorflow as tf
import socket,_thread
import os,base64
import os.path as osp
from lib.networks.factory import get_network
from demoweb import getRes_Img

from spyne import Application, rpc, ServiceBase
from spyne.protocol.soap import Soap11
from spyne.server.wsgi import WsgiApplication
from wsgiref.simple_server import make_server
from Arrdefine import *

this_dir = osp.dirname(__file__)

model=osp.join(this_dir,'model-6.16.ckpt')

if model == ' ' or not os.path.exists(model):
    print  'current path is ' + os.path.abspath(__file__)
    raise IOError(('Error: Model not found.\n'))

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
net = get_network('VGGnet_test')
saver = tf.train.Saver()
saver.restore(sess, model)

class ImgReconServices(ServiceBase):

    @rpc(Array(Req_Image), _returns=Array(Res_Image))
    def imgRecon(self, Req_Images):  #Req_Images包含ID，内容base64,需要类型（没有时是‘’）
        res_img = []  # 返回所有给定图片的检测结果，是一个数组

        n=len(Req_Images)
        for i in range(n):
            img = getRes_Img(sess, net, Req_Images[i])  # 每张图片的检测结果
            res_img.append(img)
        return res_img

#为线程定义一个函数,模型和配置文件更新，覆盖原文件,
def startthread(num):
    server1 = socket.socket()
    server1.bind(('', 8001))  # 绑定监听端口
    server1.listen(500)  # 监听
    #接收数据
    while True:
        conn, addr = server1.accept()
        os.rename(osp.join(this_dir, 'config.txt'),'configold.txt')
        os.rename(osp.join(this_dir, 'model.ckpt'),'modelold.ckpt')
        oldcfg=osp.join(this_dir, 'configold.txt')
        oldmodel = osp.join(this_dir, 'modelold.ckpt')
        newconfig = osp.join(this_dir, 'config.txt')
        newmodel = osp.join(this_dir, 'model.ckpt')

        try:
            a = ''
            while True:
                data1 = conn.recv(1)
                if data1=='':
                    break
                if (data1 == '\0'):
                    break
                else:
                    a = a + data1

            b = int(a)
            print b
            data = conn.recv(b)

            with open(newconfig, 'a') as f:
                f.write(data)

            while True:
                data = conn.recv(1024 * 1024)  # 接收
                with open(newmodel, 'ab') as f1:
                    f1.write(data)
                if not data:  # 客户端已断开
                    print '文件已传输完'
                    break

            try:
                saver.restore(sess, newmodel)
                if osp.exists(oldcfg):
                    os.remove(oldcfg)
                if osp.exists(oldmodel):
                    os.remove(oldmodel)

            except:
                if osp.exists(newconfig):
                    os.remove(newconfig)
                if osp.exists(newconfig):
                    os.remove(newmodel)
                os.rename(oldcfg,'config.txt')
                os.rename(oldmodel, 'model.ckpt')
                saver.restore(sess, osp.join(this_dir, 'model.ckpt'))


            conn.send('1')
            conn.close()

        except:
            if osp.exists(newconfig):
                os.remove(newconfig)
            if osp.exists(newconfig):
                os.remove(newmodel)
            os.rename(oldcfg, 'config.txt')
            os.rename(oldmodel, 'model.ckpt')
            saver.restore(sess, osp.join(this_dir, 'model.ckpt'))

            conn.send('0')
            conn.close()


if __name__ == "__main__":

    # 创建线程
    try:
        model=_thread.start_new_thread(startthread, (1,))
    except:
        print "Error: unable to start thread"

    soap_app = Application([ImgReconServices],
                           'ImgtestServices',
                           # in_protocol=Soap11(validator="lxml"),
                           in_protocol=Soap11(),
                           out_protocol=Soap11())
    # wsgi_app = WsgiApplication(soap_app,True,600 * 1024 * 1024)
    wsgi_app = WsgiApplication(soap_app)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('spyne.protocol.xml').setLevel(logging.DEBUG)


    server = make_server('', 8000, wsgi_app)
    server.serve_forever()
