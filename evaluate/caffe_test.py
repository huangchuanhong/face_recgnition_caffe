import numpy as np
import cv2
import sys,os
sys.path.append('../ssd_caffe/python')
from sklearn import preprocessing
import caffe
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prototxt_file', type=str)
    parser.add_argument('caffemodel_file', type=str)
    parser.add_argument('list_file', type=str)
    return parser.parse_args()

args = parse_args()


deploy = args.prototxt_file 
model = args.caffemodel_file 

np.set_printoptions(threshold=np.inf)
caffe.set_mode_gpu()
caffe.set_device(1)
net = caffe.Net(deploy,model,caffe.TEST)
mean = np.array((103.0,116.0,123.0))

root_dir = ''
filetxt = args.list_file
g = open("testerror","w")
with open(filetxt) as f:
    lines = f.readlines()
for line in lines:
    img = os.path.join(root_dir,line.strip().split()[0])
    im = cv2.imread(img)
    im = cv2.resize(im,(108,108))
    im = (im - mean) / 100.0
    im = im[...,::-1]
    im = np.transpose(im,(2,0,1))
    im = im[np.newaxis,:]
    net.blobs['blob0'].data[...] = im
    out = net.forward()
    embedding = out['fc']
    np.savetxt(g, embedding.reshape(1,-1))
g.close()
