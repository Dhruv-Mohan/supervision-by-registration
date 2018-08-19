from __future__ import division
import caffe2.python.onnx.backend as backend
#import caffe2.python.onnx as onnx

import onnx

from pathlib import Path
import sys

lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
from datasets import GeneralDataset as Dataset
from xvision import transforms
print('ineval')
# Load the ONNX model
mean_fill = tuple([int(x * 255) for x in [0.485, 0.456, 0.406]])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

model = onnx.load("cpm_vgg16-epoch-008-050.onnx")
pre_crop_expand=0.2
# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))
#rep = backend.prepare(model, device="CUDA:0")

eval_transform = transforms.Compose(
        [transforms.PreCrop(pre_crop_expand), transforms.TrainScale2WH((256, 256)),
         transforms.ToTensor(), normalize])
dataset = Dataset(eval_transform, 4.0, 8, 'gaussian', '300W-68')