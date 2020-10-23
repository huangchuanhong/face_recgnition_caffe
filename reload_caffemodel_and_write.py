import sys
#sys.path.insert(0,'../caffe/python')
sys.path.append('./ssd_caffe/python')
#sys.path.insert(0,"../../../caffe/ssd/caffe/python")
import caffe
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prototxt_file', type=str)
    parser.add_argument('caffemodel_file', type=str)
    return parser.parse_args()

args = parse_args()

caffemodel_file = args.caffemodel_file
prototxt_file = args.prototxt_file


net = caffe.Net(prototxt_file, caffemodel_file, caffe.TEST)
net.save(caffemodel_file)
