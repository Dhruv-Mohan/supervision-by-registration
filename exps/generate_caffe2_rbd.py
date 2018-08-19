
import argparse
import  numpy as np

import os
import cv2

from Records.utils.terminal_utils import progressbar
from Records.utils.pointIO import *
from Records.utils.draw import draw_pts
from Records.Collection_engine import Collection_engine

# _image_path = '/home/dhruv/Projects/Datasets/300VW_Dataset_2015_12_14/001/out/'
import caffe2.python.onnx.backend as backend
import onnx
print('import')

print('import')
model = onnx.load("cpm_vgg16-epoch-009-050.onnx")
print('import')
# Check that the IR is well formed
onnx.checker.check_model(model)
# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

_image_path = 'Menpo51220/val/'
_output_path = 'sbr/'
_pts_path_ = 'Menpo51220/pts/'
images = os.listdir(_image_path)

def normalize(im):
    im = im.astype(np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        im[:,:,i] -= mean[i]
        im[:,:,i] /= std[i]

    im = cv2.resize(im, (256, 256), 0)
    im = np.transpose(im, (2,0,1))
    im = np.expand_dims(im, 0)

    return im

def evaluate(args):
    print('Prepare input data')

    rep = backend.prepare(model, device="CUDA:0")  # or "CPU"
    print('Prepare input data')
    l1 = []
    record_writer = Collection_engine.produce_generator()
    total_images = len(images)
    for im_ind, aimage in enumerate(images):
        progressbar(im_ind, total_images)
        pts_name = os.path.splitext(aimage)[0] + '.pts'
        pts_full = _pts_path_ + pts_name
        gtpts = get_pts(pts_full, 90)
        aim = _image_path + aimage
        args.image = aim
        aim = args.image
        im = cv2.imread(aim)
        imshape = im.shape
        args.face = [0, 0, imshape[0], imshape[1]]
        image = normalize(im)
        # network forward
        c_locs, c_scors = rep.run(image)
        # obtain the locations on the image in the orignial size



        c_locations = c_locs[0, :-1, :]
        c_locations[:, 0], c_locations[:, 1] = c_locations[:, 0] * imshape[1]/256. , c_locations[:, 1] * imshape[0]/256.

        c_scores = np.expand_dims(c_scors[0, :-1], -1)

        c_pred_pts = np.concatenate((c_locations, c_scores), axis=1).transpose(1, 0)

        c_pred_pts = np.transpose(c_pred_pts, [1, 0])
        c_pred_pts = c_pred_pts[:, :-1]
        print(aim)
        print(c_pred_pts)
        input('crap')
        record_writer.consume_data(im, gt_pts=gtpts, pred_pts=c_pred_pts, name=aimage)

    record_writer.post_process()
    record_writer.generate_output(output_path=_output_path,
                                  epochs=9,
                                  name='Supervision By Registration')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a images by the trained model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = parser.parse_args()
    evaluate(args)