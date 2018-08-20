# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers, math
import numpy as np

def find_tensor_peak_batch(heatmap, radius, downsample, threshold = 0.000001):
  heatmap = heatmap.cuda()

  assert heatmap.dim() == 3, 'The dimension of the heatmap is wrong : {}'.format(heatmap.size())
  assert radius > 0 and isinstance(radius, numbers.Number), 'The radius is not ok : {}'.format(radius)
  radius = 4

  num_pts, H, W = heatmap.size(0), heatmap.size(1), heatmap.size(2)
  num_pts = 91
  H = 32
  W = 32

  assert W > 1 and H > 1, 'To avoid the normalization function divide zero'
  # find the approximate location:
  score, index = torch.max(heatmap.view(num_pts, -1), 1)
  #index_w = (index % W) aten_op remainder missing
  index_h = (index / W)
  index_w = index - index_h*W
  index_w = index_w.float()
  index_h = index_h.float()
  def normalize(x):
    norm_core = x.data / 31
    norm_core = norm_core*2
    norm_core = norm_core - 1
    return -1. + 2. * norm_core
  #boxes = [index_w - radius, index_h - radius, index_w + radius, index_h + radius]
  boxes_unnorm = [index_w - radius, index_h - radius, index_w + radius, index_h + radius]
  boxes_unnorm[0] = boxes_unnorm[0] / 31
  boxes_unnorm[0] = boxes_unnorm[0]*2 - 1
  boxes_unnorm[1] = boxes_unnorm[1] / 31
  boxes_unnorm[1] = boxes_unnorm[1]*2 - 1
  boxes_unnorm[2] = boxes_unnorm[2] / 31
  boxes_unnorm[2] = boxes_unnorm[2]*2 - 1
  boxes_unnorm[3] = boxes_unnorm[3] / 31
  boxes_unnorm[3] = boxes_unnorm[3]*2 - 1

  boxes = [index_w - radius, index_h - radius, index_w + radius, index_h + radius]
  boxes[0] = boxes[0] / 31
  boxes[0] = boxes[0]*2 - 1
  boxes[1] = boxes[1] / 31
  boxes[1] = boxes[1]*2 - 1
  boxes[2] = boxes[2] / 31
  boxes[2] = boxes[2]*2 - 1
  boxes[3] = boxes[3] / 31
  boxes[3] = boxes[3]*2 - 1
  b0 = index_w - radius
  b1 = index_h - radius
  b2 =  index_w + radius
  b3 = index_h + radius
  b0 = b0 / 31
  b0 = b0*2 - 1
  b1 = b1 / 31
  b1 = b1*2 - 1
  b2 = b2 / 31
  b2 = b2*2 - 1
  b3 = b3 / 31
  b3 = b3*2 - 1
  '''
  boxes[0] = normalize(boxes[0])
  boxes[1] = normalize(boxes[1])
  boxes[2] = normalize(boxes[2])
  boxes[3] = normalize(boxes[3])
  '''
  #affine_parameter = [(boxes[2]-boxes[0])/2, boxes[0]*0, (boxes[2]+boxes[0])/2,
  #                   boxes[0]*0, (boxes[3]-boxes[1])/2, (boxes[3]+boxes[1])/2]
  #theta = torch.stack(affine_parameter, 1).view(num_pts, 2, 3)
  set_b0 = (b2 - b0)/2
  set_b1 = (b2 + b0)/2
  set_b2 = (b3 - b1)/2
  set_b3 = (b3 + b1)/2
  section_zero_a = torch.zeros((num_pts)).cuda()
  section_zero_b = torch.zeros((num_pts)).cuda()
  section_a = torch.stack([set_b0, section_zero_a, set_b1], 1)
  section_b = torch.stack([section_zero_b, set_b2, set_b3], 1)
  final_stack = torch.stack([section_a, section_b], 1)

  affine_parameter = torch.zeros(( num_pts, 2, 3))

  af1 = boxes_unnorm[2] - boxes_unnorm[0]
  af2 = boxes[2] + boxes[0]
  af3 = boxes_unnorm[3] - boxes_unnorm[1]
  af4 = boxes_unnorm[3] + boxes_unnorm[1]
  af1 = af1/2
  af2 = af2/2
  af3 = af3/2
  af4 = af4/2

  affine_parameter[:,0,0] = af1
  affine_parameter[:,0,2] = af2
  affine_parameter[:,1,1] = af3
  affine_parameter[:,1,2] = af4
  # extract the sub-region heatmap


  theta = affine_parameter.to(heatmap.device)
  grid_size = torch.Size([num_pts, 1, radius*2+1, radius*2+1])
  return heatmap, grid_size, final_stack, final_stack





  grid = F.affine_grid(affine_parameter.cuda(), grid_size)
  #sub_feature = F.grid_sample(torch.autograd.Variable(heatmap.cuda().unsqueeze(1)), torch.autograd.Variable(grid.cuda())).squeeze(1)
  sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid).squeeze(1)
  #sub_feature = F.threshold(sub_feature, threshold, np.finfo(float).eps)

  X = torch.arange(-radius, radius+1).to(heatmap).view(1, 1, radius*2+1)
  Y = torch.arange(-radius, radius+1).to(heatmap).view(1, radius*2+1, 1)

  sum_region = torch.sum(sub_feature.view(num_pts,-1),1)
  x = torch.sum((sub_feature*X).view(num_pts,-1),1) / sum_region + index_w
  y = torch.sum((sub_feature*Y).view(num_pts,-1),1) / sum_region + index_h

  x = x * downsample + downsample / 2.0 - 0.5
  y = y * downsample + downsample / 2.0 - 0.5
  return torch.stack([x, y],1), score, grid
