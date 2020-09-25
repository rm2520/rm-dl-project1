from __future__ import absolute_import

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torchvision import models
import os
import sys
from torchvision.models.resnet import resnet50, Bottleneck
import copy

this_path = os.path.split(__file__)[0]
sys.path.append(this_path)
import senet

__all__ = ['Resenetglobal']

class Resenetglobal(nn.Module):
    def __init__(self, num_classes,part1,part2 ,loss={'xent'}, stride=2, droprate=0.5, **kwargs):
        super(Resenetglobal, self).__init__()

        self.loss = loss
        modelft = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            modelft.conv1,
            modelft.bn1,
            modelft.relu,
            modelft.maxpool,
            modelft.layer1,
            modelft.layer2,
            modelft.layer3[0],
        )

        res_conv4 = nn.Sequential(*modelft.layer3[1:])
        modelft.layer4[0].downsample[0].stride = (1, 1)
        modelft.layer4[0].conv2.stride = (1, 1)
        #res_g_conv5 = modelft.layer4
        self.part1 = part1
        self.part2 = part2
        self.avgpoolpart1 = nn.AdaptiveAvgPool2d((self.part1,1))
        self.avgpoolpart2 = nn.AdaptiveAvgPool2d(( self.part2,1))


        modelft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool =modelft.avgpool

        #self.gbranch_1st = nn.Sequential(copy.deepcopy(firstlayer))
        #self.pbranch1_1st = nn.Sequential(copy.deepcopy(firstlayer))
        #self.pbranch2_1st = nn.Sequential(copy.deepcopy(firstlayer))


        self.gbranch = nn.Sequential(copy.deepcopy(res_conv4))
        self.pbranch1 = nn.Sequential(copy.deepcopy(res_conv4))
        self.pbranch2 = nn.Sequential(copy.deepcopy(res_conv4))


        self.feature_dim = 1024
        self.norm1 = nn.BatchNorm2d(self.feature_dim)
        #self.norm2 = nn.LayerNorm(288)

        #self.ff = nn.Sequential(
         #   nn.Linear(288, 2*288),
          #  nn.ReLU(),
           # nn.Linear(2 * 288, 288))



        #self.attention = Temporalpooling(self.feature_dim, num_classes, droprate=0.5, w=1, h=1)


    def global_branch_fn(self,x,l=0):

        x = self.gbranch(x)
       # x = self.avgpool(x)
        return x
    def local_branch_fn1(self,x):
        x = self.pbranch1(x)
        parts = self.avgpoolpart1(x)
        return x,parts
    def local_branch_fn2(self,x):
        x = self.pbranch2(x)
        att=self.avgpoolpart1(x)
        parts = self.avgpoolpart2(x)
       # x = self.avgpool(x)
        return x,parts,att
    def spatial_attention(self,x,b,t):
        xp1 ,parts1= self.local_branch_fn1(x)

        xp2 ,parts2,att= self.local_branch_fn2(x)
        att=att.permute(0,1,3,2)
        parts=parts1@att
        parts1=parts1.permute(0,1,3,2)

        parts1=parts1+parts1@parts
        parts1=parts1.permute(0,1,3,2)

        h=xp1.size(2)
        w=xp1.size(3)
        k=xp1.size(1)
        xp1 = xp1.view(xp1.size(0), xp1.size(1), h * w )
        xp1=xp1.permute(0,2,1)
        xp2 = xp2.view(xp2.size(0), xp2.size(1), h * w)

        x=xp1 @ xp2
        xp1=xp1.permute(0,2,1)
        x=F.softmax(x, dim=-1)
        xp1 = xp1.view(xp1.size(0), xp1.size(1), h , w)
        xp2 = xp2.view(xp2.size(0), xp2.size(1), h , w)
        xp1=self.avgpool(xp1)
        xp2 = self.avgpool(xp2)
        xp1 = torch.squeeze(xp1, 3)
        xp2 = torch.squeeze(xp2, 3)
        xp2=xp2.permute(0,2,1)
        a=xp1@xp2
        a=F.softmax(a, dim=-1)





        return x,a,parts1,parts2



    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.backbone(x)
        alpha,a,parts1,parts2=self.spatial_attention(x,b,t)
        x=self.global_branch_fn(x)
        w = x.size(3)
        h = x.size(2)
        x = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))
        x = x + x @ alpha






        x = x.view(x.size(0), x.size(1), h, w)

        x=self.avgpool(x)
        x = torch.squeeze(x, 3)
        x = x.permute(0, 2, 1)
        x = x + x @ a
        x = x.permute(0, 2, 1)


        x = torch.unsqueeze(x, 3)












        if not self.training:
            return x,parts1,parts2

        if self.loss == {'xent'}:
            return x,parts1,parts2
        elif self.loss == {'xent', 'htri'}:
            return x,parts1,parts2
        elif self.loss == {'cent'}:
            return x,parts1,parts2
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))







