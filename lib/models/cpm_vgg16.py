# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division
import time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from copy import deepcopy
from .model_utils import get_parameters
from .basic_batch import find_tensor_peak_batch
from .initialization import weights_init_cpm
from torch.autograd import Variable, Function
import pickle

class VGG16_base(nn.Module):
  def __init__(self, config, pts_num):
    super(VGG16_base, self).__init__()

    self.config = deepcopy(config)
    self.downsample = 8
    self.pts_num = pts_num

    self.features = nn.Sequential(
          nn.Conv2d(  3,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d( 64,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d( 64, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(256, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True))
  

    self.CPM_feature = nn.Sequential(
          nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), #CPM_1
          nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True)) #CPM_2

    assert self.config.stages >= 1, 'stages of cpm must >= 1 not : {:}'.format(self.config.stages)
    stage1 = nn.Sequential(
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 512, kernel_size=1, padding=0), nn.ReLU(inplace=True),
          nn.Conv2d(512, pts_num, kernel_size=1, padding=0))
    stages = [stage1]
    for i in range(1, self.config.stages):
      stagex = nn.Sequential(
          nn.Conv2d(128+pts_num, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=1, padding=0), nn.ReLU(inplace=True),
          nn.Conv2d(128,     pts_num, kernel_size=1, padding=0))
      stages.append( stagex )
    self.stages = nn.ModuleList(stages)
  
  def specify_parameter(self, base_lr, base_weight_decay):
    params_dict = [ {'params': get_parameters(self.features,   bias=False), 'lr': base_lr  , 'weight_decay': base_weight_decay},
                    {'params': get_parameters(self.features,   bias=True ), 'lr': base_lr*2, 'weight_decay': 0},
                    {'params': get_parameters(self.CPM_feature, bias=False), 'lr': base_lr  , 'weight_decay': base_weight_decay},
                    {'params': get_parameters(self.CPM_feature, bias=True ), 'lr': base_lr*2, 'weight_decay': 0},
                  ]
    for stage in self.stages:
      params_dict.append( {'params': get_parameters(stage, bias=False), 'lr': base_lr*4, 'weight_decay': base_weight_decay} )
      params_dict.append( {'params': get_parameters(stage, bias=True ), 'lr': base_lr*8, 'weight_decay': 0} )
    return params_dict

  # return : cpm-stages, locations
  def forward(self, inputs):
    assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
    batch_size, feature_dim = inputs.size(0), inputs.size(1)
    batch_cpms, batch_locs, batch_scos = [], [], []

    feature  = self.features(inputs)
    xfeature = self.CPM_feature(feature)
    for i in range(self.config.stages):
      if i == 0: cpm = self.stages[i]( xfeature )
      else:      cpm = self.stages[i]( torch.cat([xfeature, batch_cpms[i-1]], 1) )
      batch_cpms.append( cpm )


    with open('aff_grid.pickle', 'rb') as crap:
        grid = pickle.load(crap)


    heatmap, grid_size, affine_parameter, boxes = find_tensor_peak_batch(batch_cpms[-1][0], self.config.argmax, self.downsample)

    fun_grid = generate_grid(affine_parameter, 91, 9, 9)

    sub_feature , grid_p, boxes = MyFunction.apply(heatmap, grid_size, affine_parameter, fun_grid.cuda(), boxes)


    '''
    with open('aff_grid.pickle', 'wb') as crap:
        pickle.dump(aff_grid, crap)
    '''
    sub_feature = sub_feature.squeeze(1)
    num_pts=91
    radius=4
    H = 32
    W = 32
    score, index = torch.max(heatmap.view(num_pts, -1), 1)
    # index_w = (index % W) aten_op remainder missing
    index_h = (index / W)
    index_w = index - index_h * W
    index_w = index_w.float()
    index_h = index_h.float()

    X = torch.arange(-radius, radius + 1).to(heatmap).view(1, 1, radius * 2 + 1)
    Y = torch.arange(-radius, radius + 1).to(heatmap).view(1, radius * 2 + 1, 1)
    sum_region = torch.sum(sub_feature.view(num_pts, -1), 1)
    x = torch.sum((sub_feature * X).view(num_pts, -1), 1) / sum_region + index_w
    y = torch.sum((sub_feature * Y).view(num_pts, -1), 1) / sum_region + index_h

    x = x * self.downsample + self.downsample / 2.0 - 0.5
    y = y * self.downsample + self.downsample / 2.0 - 0.5
    locs = torch.stack([x, y],1)



    return locs, score, boxes


    # The location of the current batch
    '''
    for ibatch in range(batch_size):
        batch_location, batch_score, heatmap = find_tensor_peak_batch(batch_cpms[-1][ibatch], self.config.argmax, self.downsample)
        batch_locs.append( batch_location )
        batch_scos.append( batch_score )
        batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)
    '''

    return batch_locs, batch_scos, fun_grid

# use vgg16 conv1_1 to conv4_4 as feature extracation        
model_urls = 'https://download.pytorch.org/models/vgg16-397923af.pth'

def cpm_vgg16(config, pts):
  
  print ('Initialize cpm-vgg16 with configure : {}'.format(config))
  model = VGG16_base(config, pts)
  model.apply(weights_init_cpm)

  if config.pretrained:
    print ('vgg16_base use pre-trained model')
    weights = model_zoo.load_url(model_urls)
    model.load_state_dict(weights, strict=False)
  return model

class MyFunction(Function):
    @staticmethod
    def forward(ctx,heatmap, grid_size, affine_parameter, grid_p, boxes):
      #grid = torch.cudnn_affine_grid_generator(affine_parameter.cuda(), 91, 1, 9, 9)
      #grid = F.affine_grid(affine_parameter.cuda(), grid_size)

      unsqueezed_heatmap=heatmap.unsqueeze(1)
      sub_feature = F.grid_sample(unsqueezed_heatmap, grid_p)
      return sub_feature,  grid_p, boxes
    @staticmethod
    def symbolic(graph, heatmap, grid_size, affine_parameter, grid_p, boxes):

      #x2 = graph.at("cudnn_affine_grid_generator", affine_parameter, N_i=91, C_i=1, H_i=9, W_i=9)
      unsqueezed_heatmap = graph.at("unsqueeze", heatmap, dim_i=1)
      r = graph.at("grid_sampler", unsqueezed_heatmap, grid_p,  padding_mode_i=0)

      # x, y, x2, and r are 'Node' objects
      # print(r) or print(graph) will print out a textual representation for debugging.
      # this representation will be converted to ONNX protobufs on export.
      return r, grid_p, boxes


def generate_grid(theta, N, H, W):

    base_grid = theta.new(N, H, W, 3)
    linear_points = torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1])
    base_grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(base_grid[:, :, :, 0])
    linear_points = torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])
    base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(base_grid[:, :, :, 1])
    base_grid[:, :, :, 2] = 1

    grid = torch.bmm(base_grid.view(N, H * W, 3), theta.transpose(1, 2))

    grid = grid.view(N, H, W, 2)

    return grid


