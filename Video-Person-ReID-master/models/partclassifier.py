import torchvision
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torchvision import models
import os
import sys
import torch





def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x


class Classifier(nn.Module):
    def __init__(self, feat_num, num_classes, drop=0.5):
        super(Classifier, self).__init__()
        self.feat_dim=feat_num
        self.classifier = ClassBlock(self.feat_dim, num_classes, droprate=drop)

    def forward(self,x):
        y = self.classifier(x)
        return y


class parts_model(nn.Module):
    def __init__(self,  Feat_dim, num_classes,part1,part2, droprate=0.5, w=1, h=1, **kwargs):
        super(parts_model, self).__init__()

        self.part1 = part1
        self.part2 = part2
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        self.feat_dim = Feat_dim  # feature dimension
        self.middle_dim = 256  # middle layer dimension
        self.globalclassifier = ClassBlock(self.feat_dim, num_classes, droprate=droprate)
        self.attention_conv = nn.Conv2d(Feat_dim, self.middle_dim, [1, 1])  # 7,4 cooresponds to 224, 112 input image size

        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)


        for i in range(self.part1):
            name = 'classifier1' + str(i)
            setattr(self, name, ClassBlock(Feat_dim, num_classes, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))
        for i in range(self.part2):
            name = 'classifier2' + str(i)
            setattr(self, name, ClassBlock(Feat_dim, num_classes, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))


    def attention(self, x, b, t):

        a = F.relu(self.attention_conv(x))

        a = a.view(b, t, self.middle_dim)
        a = a.permute(0, 2, 1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        x = F.avg_pool2d(x, x.size()[2:])
        if self.att_gen == 'softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else:
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        x = x.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)

        f = att_x.view(b, self.feat_dim)
        #y = self.classifier(f)
        return f
    def forward(self, x , p1,p2, b, t):

        fgl=self.attention(x,b,t)
        ygl=self.globalclassifier(fgl)
        f = []
        y = []
        # get six part feature batchsize*2048*6
        if self.training:
            for i in range(self.part1):
                part = torch.unsqueeze(p1[:, :,i], 3)
                #part = p[:, :, :,i]
                f.append(self.attention(part, b, t))
                name = 'classifier1' + str(i)
                c = getattr(self, name)
                y.append(c(f[i]))
            for i in range(self.part2):
                part = torch.unsqueeze(p2[:, :, i], 3)
                # part = p[:, :, :,i]
                f.append(self.attention(part, b, t))
                name = 'classifier2' + str(i)
                c = getattr(self, name)
                y.append(c(f[i]))

            return ygl,fgl,y, f

        else:
            f.append(torch.unsqueeze(fgl, 2))
            for i in range(self.part1):
                part = torch.unsqueeze(p1[:, :, i], 3)
                f.append(torch.unsqueeze(self.attention(part, b, t),2))
            for i in range(self.part2):
                part = torch.unsqueeze(p2[:, :, i], 3)
                f.append(torch.unsqueeze(self.attention(part, b, t), 2))

            f = torch.cat(f, 2)
            return f