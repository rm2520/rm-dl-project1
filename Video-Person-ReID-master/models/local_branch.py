from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn.init as init
import torchvision
class Local_branch(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(Local_branch, self).__init__()
        self.num_classes = num_classes
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.base1 = self.base[0:4]
        self.feat_dim = 2048  # feature dimension




    def forward(self, x,b,t):
        x = self.base1(x)
        part=torch.split(x,8,2)
        part=list(part)
        p=len(part)
        for i in range(p):

            part[i]= self.local_layer(part[i],b,t)

        part=torch.cat(part,2)
        part = F.avg_pool1d(part, p)
        part = part.view(b, self.feat_dim)

        return part


    def local_layer(self, x,b,t):
        x = self.base[4](x)
        x = self.base[5](x)
        x = self.base[6](x)
        x = self.base[7](x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        x = x.permute(0, 2, 1)
        x = F.avg_pool1d(x, t)
        return x





