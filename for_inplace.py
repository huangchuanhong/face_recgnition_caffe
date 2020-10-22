import sys
#sys.path.insert(0,'../caffe/python')
sys.path.append('../ssd_caffe/python')
#sys.path.insert(0,"../../../caffe/ssd/caffe/python")
import caffe
from caffe.proto import caffe_pb2
from caffe.proto import caffe_pb2
zfmodel = caffe_pb2.NetParameter()
#caffemodel = './cpu_cropface_v3_inference.caffemodel'

caffemodel = sys.argv[2]#'group_kd_v4_inference.caffemodel'#./cpu_cropface_v3_inference.caffemodel'
# caffemodel = 'new.caffemodel'
f = open(caffemodel,"rb")
zfmodel.ParseFromString(f.read())
print(dir(zfmodel.layers))
f.close()
name_dict={}
cut = []
for i in range(1,len(zfmodel.layer)):
	tp = zfmodel.layer[i].type
	if tp == 'Scale' or tp == 'BatchNorm':
		cut.append(i)
	if tp == 'PReLU' or tp == 'ReLU':
		name_dict[zfmodel.layer[i].top[0]] = zfmodel.layer[i].bottom[0]
		zfmodel.layer[i].top[0] = zfmodel.layer[i].bottom[0]
	bottom = zfmodel.layer[i].bottom[0]
	if name_dict.get(bottom) != None:
		bottom = name_dict[bottom]
nums = len(cut)
#print(cut)
#print(name_dict)
print('dir(zfmodel.layer)=', dir(zfmodel.layer))
for i in range(nums):
	cut_layer = cut.pop(-1)
	zfmodel.layer.pop(cut_layer)
new_caffemodel=caffemodel.split(".")[0]+("_inplace.caffemodel")
print(new_caffemodel)
with open(new_caffemodel,"wb") as g:
	g.write(zfmodel.SerializeToString())
prototxt = sys.argv[1]#'group_kd_v4_inference.prototxt'#./cpu_cropface_v3_inference.prototxt'
#prototxt = './cpu_cropface_v3_inference.prototxt'
with open(prototxt) as f:
	lines = f.readlines()
new_protoxt = prototxt.split(".")[0]+("_inplace.prototxt")	
with open(new_protoxt,"w") as g:
	for line in lines:
		for key in name_dict.keys():
			a = line.split('"')
			# print(a)
			for c in a:
				if key == c:
					print(line)
					line = line.replace(key,name_dict[key])
		g.write(line)


# print(zfmodel)
# print(dir(zfmodel))
# g = open('new.caffemodel',"wb")
# g.write(zfmodel.SerializeToString())
# print(zfmodel.SerializeToString())
