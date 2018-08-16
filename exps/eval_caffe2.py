from __future__ import division

import sys, time, torch, random, argparse, PIL
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
import cv2

from Records.utils.terminal_utils import progressbar
from Records.utils.pointIO import *
from Records.utils.draw import draw_pts
import natsort
import json
_image_path = '/home/dhruv/Projects/Datasets/Groomyfy_16k/Dlibdet/val/'
_output_path = '/home/dhruv/Projects/TFmodels/sbr/'
_pts_path_ = '/home/dhruv/Projects/Datasets/Groomyfy_16k/Dlibdet/pts/'
# _image_path = '/home/dhruv/Projects/Datasets/300VW_Dataset_2015_12_14/001/out/'
import onnx
import caffe2.python.onnx.backend as backend
model = onnx.load("cpm_vgg16-epoch-009-050.onnx")
# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

def evaluate(args):
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


    print('The model is {:}'.format(args.model))
    snapshot = Path(args.model)
    assert snapshot.exists(), 'The model path {:} does not exist'
    snapshot = torch.load(snapshot)

    # General Data Argumentation
    mean_fill = tuple([int(x * 255) for x in [0.485, 0.456, 0.406]])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    param = snapshot['args']
    eval_transform = transforms.Compose(
        [transforms.PreCrop(param.pre_crop_expand), transforms.TrainScale2WH((param.crop_width, param.crop_height)),
         transforms.ToTensor(), normalize])
    model_config = load_configure(param.model_config, None)
    dataset = Dataset(eval_transform, param.sigma, model_config.downsample, param.heatmap_type, param.data_indicator)
    dataset.reset(param.num_pts)

    net = obtain_model(model_config, param.num_pts + 1)
    net = net.cuda()
    weights = remove_module_dict(snapshot['state_dict'])
    nu_weights = {}
    for key, val in weights.items():
        nu_weights[key.split('detector.')[-1]] = val
        print(key.split('detector.')[-1])
    weights = nu_weights
    net.load_state_dict(weights)

    print('Prepare input data')
    images = os.listdir(args.image_path)
    images = natsort.natsorted(images)
    total_images = len(images)
    rep = backend.prepare(model, device="CUDA:0")  # or "CPU"
    for im_ind, aimage in enumerate(images):
        progressbar(im_ind, total_images)
        aim = os.path.join(args.image_path, aimage)
        args.image = aim
        im = cv2.imread(aim)
        imshape = im.shape
        args.face = [0, 0, imshape[0], imshape[1]]
        [image, _, _, _, _, _, cropped_size], meta = dataset.prepare_input(args.image, args.face)
        inputs = image.unsqueeze(0).cuda()
        # network forward
        with torch.no_grad():
            batch_locs, batch_scos = net(inputs)
            c_im = np.expand_dims(image.data.numpy(), 0)
            c_locs, c_scors = rep.run(c_im)
        # obtain the locations on the image in the orignial size
        cpu = torch.device('cpu')
        np_batch_locs, np_batch_scos, cropped_size = batch_locs.to(cpu).numpy(), batch_scos.to(
            cpu).numpy(), cropped_size.numpy()
        locations, scores = np_batch_locs[0, :-1, :], np.expand_dims(np_batch_scos[0, :-1], -1)

        scale_h, scale_w = cropped_size[0] * 1. / inputs.size(-2), cropped_size[1] * 1. / inputs.size(-1)

        locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + cropped_size[2], locations[:, 1] * scale_h + \
                                           cropped_size[3]
        prediction = np.concatenate((locations, scores), axis=1).transpose(1, 0)

        c_locations = c_locs[0, :-1, :]
        c_locations[:, 0], c_locations[:, 1] = c_locations[:, 0] * scale_w + cropped_size[2], c_locations[:, 1] * scale_h + \
                                           cropped_size[3]
        c_scores = np.expand_dims(c_scors[0, :-1], -1)

        c_pred_pts = np.concatenate((c_locations, c_scores), axis=1).transpose(1, 0)

        pred_pts = np.transpose(prediction, [1, 0])
        pred_pts = pred_pts[:, :-1]


        c_pred_pts = np.transpose(c_pred_pts, [1, 0])
        c_pred_pts = c_pred_pts[:, :-1]
        print(c_scors,'\n\n\n')
        print(np_batch_scos)
        print(c_scors - np_batch_scos)

        if args.save:
            json_file = os.path.splitext(aimage)[0] + '.jpg'
            save_path = os.path.join(args.save, 'caf' + json_file)
            save_path2 = os.path.join(args.save, 'py_'+ json_file)

            sim2 = draw_pts(im, pred_pts=pred_pts, get_l1e=False)
            sim = draw_pts(im, pred_pts=c_pred_pts, get_l1e=False)
            #print(pred_pts)
            cv2.imwrite(save_path, sim)
            cv2.imwrite(save_path2, sim2)
            input('save1')
            # image.save(args.save)
            # print ('save the visualization results into {:}'.format(args.save))

        else:
            print('ignore the visualization procedure')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a images by the trained model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, help='The snapshot to the saved detector.')
    parser.add_argument('--save', type=str, help='The path to save the visualized results.')
    parser.add_argument('--image_path', type=str, help='The path to load images from.')
    args = parser.parse_args()
    evaluate(args)