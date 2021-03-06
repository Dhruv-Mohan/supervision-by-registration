
from __future__ import division


import sys, time, torch, random, argparse, PIL
torch.set_default_tensor_type(torch.FloatTensor)
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path
import numbers, numpy as np

lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
from datasets import GeneralDataset as Dataset
from xvision import transforms, draw_image_by_points
from models import obtain_model, remove_module_dict
from config_utils import load_configure
import os
import torch.onnx
import cv2
from Records.utils.draw import draw_pts
#import onnx
#import caffe2.python.onnx.backend as backend
from Records.utils.terminal_utils import progressbar
from Records.Collection_engine import Collection_engine
from Records.utils.pointIO import *
from torch.autograd import Variable

_image_path = 'Menpo51220/val/'
_output_path = 'sbr/'
_pts_path_ = 'Menpo51220/pts/'
#_image_path = '/home/dhruv/Projects/Datasets/300VW_Dataset_2015_12_14/001/out/'
images = os.listdir(_image_path)
import pickle


def evaluate(args):
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    model_name = os.path.split(args.model)[-1]
    onnx_name = os.path.splitext(model_name)[0] + ".onnx"

    print('The model is {:}'.format(args.model))
    print('Model name is {:} \nOutput onnx file is {:}'.format(model_name, onnx_name))

    snapshot = Path(args.model)
    assert snapshot.exists(), 'The model does not exist {:}'
    #print('Output onnx file is {:}'.format(onnx_name))
    snapshot = torch.load(snapshot)

    # General Data Argumentation
    mean_fill = tuple([int(x * 255) for x in [0.485, 0.456, 0.406]])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    param = snapshot['args']
    print(param)

    eval_transform = transforms.Compose(
        [transforms.PreCrop(param.pre_crop_expand), transforms.TrainScale2WH((param.crop_width, param.crop_height)),
         transforms.ToTensor(), normalize])
    model_config = load_configure(param.model_config, None)
    print(model_config)

    dataset = Dataset(eval_transform, param.sigma, model_config.downsample, param.heatmap_type, param.data_indicator)
    dataset.reset(param.num_pts)


    net = obtain_model(model_config, param.num_pts + 1)
    net = net
    weights = remove_module_dict(snapshot['state_dict'])

    nu_weights = {}
    for key, val in weights.items():
        nu_weights[key.split('detector.')[-1]] = val
        print(key.split('detector.')[-1])
    weights = nu_weights

    net.load_state_dict(weights)


    input_name = ['image_in']
    output_name = ['locs', 'scors', 'crap']


    im = cv2.imread('Menpo51220/val/0000018.jpg')

    imshape = im.shape
    face = [0, 0, imshape[0], imshape[1]]
    [image, _, _, _, _, _, cropped_size], meta = dataset.prepare_input('Menpo51220/val/0000018.jpg', face)
    dummy_input = torch.randn(1, 3, 256, 256, requires_grad=True, dtype= torch.float32)
    input(dummy_input.dtype)
    #input('imcrap')



    inputs = image.unsqueeze(0)
    out_in = inputs.data.numpy()
    with open('pick.pick' , 'wb') as crap:
        pickle.dump(out_in, crap)

    with torch.no_grad():
        batch_locs, batch_scos, heatmap= net(inputs)
        torch.onnx.export(net.cuda(), dummy_input.cuda(), onnx_name, verbose=True, input_names=input_name, output_names=output_name, export_params=True)
        print(batch_locs)
        print(batch_scos)
        print(heatmap)
    cpu = torch.device('cpu')
    np_batch_locs, np_batch_scos, cropped_size = batch_locs.to(cpu).numpy(), batch_scos.to(cpu).numpy(), cropped_size.numpy()
    locations = np_batch_locs[:-1,:]
    scores = np.expand_dims(np_batch_scos[:-1], -1)

    scale_h, scale_w = cropped_size[0] * 1. / inputs.size(-2) , cropped_size[1] * 1. / inputs.size(-1)

    locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w, locations[:, 1] * scale_h
    prediction = np.concatenate((locations, scores), axis=1).transpose(1,0)


    pred_pts = np.transpose(prediction, [1, 0])
    pred_pts = pred_pts[:, :-1]
    #print(pred_pts)
    sim = draw_pts(im, pred_pts=pred_pts, get_l1e=False)
    cv2.imwrite('py_0.jpg', sim)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export trained model to onnx',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, help='The snapshot to the saved detector.')
    args = parser.parse_args()
    evaluate(args)