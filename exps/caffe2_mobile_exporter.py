
import argparse
import  numpy as np

import os
import cv2

from Records.utils.terminal_utils import progressbar
from Records.utils.pointIO import *
from Records.utils.draw import draw_pts

# _image_path = '/home/dhruv/Projects/Datasets/300VW_Dataset_2015_12_14/001/out/'
import caffe2.python.onnx.backend as backend
from caffe2.python.predictor import mobile_exporter
import onnx
print('import')

print('import')
model = onnx.load("cpm_vgg16-epoch-009-050.onnx")
print('import')
# Check that the IR is well formed
onnx.checker.check_model(model)
# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))
import pickle
with open('pick.pick', 'rb') as crap:
    in_fi = pickle.load(crap)


def normalize(im):
    im = im.astype(np.float32)/255.
    im = cv2.resize(im, (256, 256), 0)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i in range(3):
        im[:,:,i] -= mean[i]
        im[:,:,i] /= std[i]


    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, 0)

    return im

def evaluate(args):
    print('Prepare input data')

    rep = backend.prepare(model, device="CUDA:0")  # or "CPU"
    prepared_backend = rep
    c2_workspace = prepared_backend.workspace
    c2_model = prepared_backend.predict_net

    init_net, predict_net = mobile_exporter.Export(c2_workspace, c2_model, c2_model.external_input)
    with open('init_net.pb', "wb") as fopen:
        fopen.write(init_net.SerializeToString())
    with open('predict_net.pb', "wb") as fopen:
        fopen.write(predict_net.SerializeToString())
    pass
    args.image = 'Menpo51220/val/0000018.jpg'
    aim = args.image
    im = cv2.imread(aim)
    imshape = im.shape
    args.face = [0, 0, imshape[0], imshape[1]]
    image = normalize(im)
    # network forward


    c_locs, c_scors, heatmap = rep.run(image)
    # obtain the locations on the image in the orignial size
    print(c_locs)
    #print(c_scors, '\n\n\n')
    print(heatmap)
    c_locations = c_locs[:-1, :]
    c_locations[:, 0], c_locations[:, 1] = c_locations[:, 0] * imshape[1]/256. , c_locations[:, 1] * imshape[0]/256.

    c_scores = np.expand_dims(c_scors[:-1], -1)
    print(c_locations.shape)
    print(c_scores.shape)
    c_pred_pts = np.concatenate((c_locations, c_scores), axis=1).transpose(1, 0)


    c_pred_pts = np.transpose(c_pred_pts, [1, 0])
    c_pred_pts = c_pred_pts[:, :-1]
    #print(c_pred_pts, '\n\n\n')
    sim = draw_pts(im, pred_pts=c_pred_pts, get_l1e=False)
    cv2.imwrite('caf_0.jpg', sim)

    if args.save:
            json_file = os.path.splitext(aimage)[0] + '.jpg'
            save_path = os.path.join(args.save, 'caf' + json_file)
            sim = draw_pts(im, pred_pts=c_pred_pts, get_l1e=False)
            #print(pred_pts)
            cv2.imwrite(save_path, sim)
            input('save1')
            # image.save(args.save)
            # print ('save the visualization results into {:}'.format(args.save))

    else:
            print('ignore the visualization procedure')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a images by the trained model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = parser.parse_args()
    evaluate(args)