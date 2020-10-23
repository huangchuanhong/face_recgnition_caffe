from __future__ import division
import numpy as np
import cv2
import sys
import os
import time
#sys.path.append('../rfcn_learn/caffe/python')
sys.path.append('./ssd_caffe/python')
import caffe
import argparse
#np.set_printoptions(threshold=np.inf)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prototxt_file', type=str)
    parser.add_argument('caffemodel_file', type=str)
    parser.add_argument('-l', '--last_top', type=str, default='fc')
    return parser.parse_args()

args = parse_args()
   
caffemodel_file = args.caffemodel_file 
prototxt_file = args.prototxt_file
#caffemodel_file = 'merged_res50_inplace_poolk2.caffemodel'
#prototxt_file = 'merged_res50_inplace_poolk2_mvglobal.prototxt'
gpu_id = 0

class FaceRecTester(object):
    def __init__(self, gpu_id, prototxt_file, caffemodel_file):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()
        self.net = caffe.Net(prototxt_file, caffemodel_file, caffe.TEST)
        #for key in self.net.params.keys():
        #    ##print(key, self.net.params[key][0].data.shape)
        #    if key == 'backbone_conv1':#'backbone_layer4_1_conv2':
        #        print(key,self.net.params[key][0].data)
        #        print(self.net.params[key][1].data)
        #        exit()
    def preprocess(self, img_file):
        mean = np.array([0.482352, 0.45490, 0.40392])
        std = np.array([0.392157, 0.392157, 0.392157])
        img = cv2.imread(img_file)
        img = img[...,::-1]
        img = cv2.resize(img, (108, 108))
        img = (img / 255.- mean) / std
        img = img.transpose((2, 0, 1))[np.newaxis, ...]
        return img

    def inference(self, img):
        self.net.blobs['blob0'].reshape(*(img.shape))
        feature = self.net.forward(blob0=img)[args.last_top]#['fc_blob185']#['fc_blob265']
        return feature
        

tester = FaceRecTester(gpu_id, prototxt_file, caffemodel_file)
#img = tester.preprocess('1542.bmp')
img = tester.preprocess('img.jpg')
#img = np.ones([1,3,108, 108], dtype=np.float32)
start = time.time()
feature = tester.inference(img)
print(feature)
with open('result.txt', 'w') as f:
    for i in feature[0]:
        f.write(str(i) + ' ')


