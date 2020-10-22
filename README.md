# face_recgnition_caffe
use after dist_face_pytorch

# merge conv_bn_relu to a single conv
```bash
python gen_merged_model.py something.prototxt something.caffemodel
```
# set ReLU inplace=True
```bash
python for_inplace.py something_inference.prototxt something_inference.caffemodel
```
# reload
```bash
python reload_caffemodel_and_write.py something_inference_inplace.prototxt something_inference_inplace.caffemodel
```
# inference
```bash
python inference.py something_inference_inplace.prototxt something_inference_inplace.caffemodel
```
# evaluate
## generate features
```bash
python caffe_test.py something_inference_inplace.prototxt something_inference_inplace.caffemodel list.txt
```
## evaluate
```bash
python face_eval_v1.py list.txt testerror
```
