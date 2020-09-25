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

__all__ = ['ResNet50TP', 'ResNet50TA', 'ResNet50RNN','TST']



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


class ResNet50TP(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TP, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        x = x.permute(0, 2, 1)
        f = F.avg_pool1d(x, t)
        f = f.view(b, self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50test(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TA, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.baselocal = (nn.Sequential(*list(resnet50.children()))[:-2])[4:8]
        self.base1 = self.base[0:4]
        self.base2 = self.base[4:8]
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048  # feature dimension
        self.middle_dim = 256  # middle layer dimension
        self.classifier = nn.Linear(self.feat_dim, num_classes)

        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim,
                                        [7, 4])  # 7,4 cooresponds to 224, 112 input image size

        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)

    '''def local_branch(self, x, b, t):
        x = torch.split(x, 8, 3)
        p = len(x)
        x = torch.cat(x, 1)
        x = x.view(b * t * p, x.size(2), x.size(3), x.size(4))
        x = self.baselocal(x)
        #x = F.avg_pool2d(x, x.size()[2:])
        x = torch.squeeze(x,2 )
        x = x.view(b *t , p, x.size(1), x.size(2))
        x = x.permute(0, 2,1, 3)

        #x = x.view(b, t, p, -1)

        #x = F.avg_pool2d(x, x.size()[2:])
        #x = x.view(b, self.feat_dim)
        #localy = self.classifier2(x)
        return  x'''

    def global_branch(self, x, b, t):
        x = self.base2(x)
        # y,f=self.attention(x,b,t)
        return x

    def attention(self, xglob, b, t):
        # x = self.avgpool(xglob+xloc)

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
        y = self.classifier(f)
        return y, f

    def get_feature(self, x, b, t):
        xglob = self.global_branch(x, b, t)
        # x = torch.unsqueeze(x, 1)
        # xloc = self.local_branch(x, b, t)
        y, f = self.attention(xglob, b, t)
        return y, f

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))

        x = self.base1(x)
        y, f = self.get_feature(x, b, t)
        if not self.training:
            return f

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TAclassblock(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, stride=2, droprate=0.5, **kwargs):
        super(ResNet50TA, self).__init__()
        self.loss = loss
        model_ft = torchvision.models.resnet50(pretrained=True)
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, num_classes, droprate)
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048  # feature dimension
        self.middle_dim = 256  # middle layer dimension
        # self.classifier = nn.Linear(self.feat_dim, num_classes)

        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim,
                                        [7, 4])  # 7,4 cooresponds to 224, 112 input image size

        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)

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
        y = self.classifier(f)
        return y, f

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # x = self.model.avgpool(x)
        # x = x.view(b, t ,x.size(1))

        y, f = self.attention(x, b, t)
        if not self.training:
            return f

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TAdense(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, stride=2, droprate=0.5, **kwargs):
        super(ResNet50TA, self).__init__()
        self.loss = loss

        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        self.classifier = ClassBlock(1024, num_classes, droprate)
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        self.feat_dim = 1024  # feature dimension
        self.middle_dim = 256  # middle layer dimension
        # self.classifier = nn.Linear(self.feat_dim, num_classes)

        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim,
                                        [1, 1])  # 7,4 cooresponds to 224, 112 input image size

        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)

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
        y = self.classifier(f)
        return y, f

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.model.features(x)

        y, f = self.attention(x, b, t)
        if not self.training:
            return f

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TAmid(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, stride=2, droprate=0.5, **kwargs):
        super(ResNet50TA, self).__init__()
        self.loss = loss

        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft

        self.classifier = ClassBlock(2048 + 1024, num_classes, droprate)
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        self.feat_dim = 1024 + 2048  # feature dimension
        self.middle_dim = 256  # middle layer dimension

        self.attention_conv = nn.Conv2d(2048 + 1024, self.middle_dim,
                                        [1, 1])  # 7,4 cooresponds to 224, 112 input image size

        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)

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
        y = self.classifier(f)
        return y, f

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0, x1), 1)

        y, f = self.attention(x, b, t)
        if not self.training:
            return f

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TA_part(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, stride=2, droprate=0.5, **kwargs):
        super(ResNet50TA, self).__init__()
        self.loss = loss

        self.part = 7  # We cut the pool5 to 6 parts
        model_ft = senet.se_resnet101(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        # self.model.layer4[0].downsample[0].stride = (1, 1)
        # self.model.layer4[0].conv2.stride = (1, 1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048  # feature dimension
        self.middle_dim = 256  # middle layer dimension

        self.attention_conv = nn.Conv2d(2048, self.middle_dim, [1, 1])  # 7,4 cooresponds to 224, 112 input image size

        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)

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
        # y = self.classifier(f)
        return f

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.model(x)

        f = []
        y = []
        # get six part feature batchsize*2048*6
        if self.training:
            for i in range(self.part):
                part = torch.unsqueeze(x[:, :, i], 3)
                f.append(self.attention(part, b, t))
                name = 'classifier' + str(i)
                c = getattr(self, name)
                y.append(c(f[i]))
        else:
            for i in range(self.part):
                part = torch.unsqueeze(x[:, :, i], 3)
                f.append(torch.unsqueeze(self.attention(part, b, t), 2))
            f = torch.cat(f, 2)

            return f

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TA(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, stride=2, droprate=0.5, **kwargs):
        super(ResNet50TA, self).__init__()
        self.loss = loss

        model_ft = senet.se_resnet50(pretrained=True)

        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft

        self.classifier = ClassBlock(2048 + 1024, num_classes, droprate)
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        self.feat_dim = 1024 + 2048  # feature dimension
        self.middle_dim = 256  # middle layer dimension

        self.attention_conv = nn.Conv2d(2048 + 1024, self.middle_dim,
                                        [1, 1])  # 7,4 cooresponds to 224, 112 input image size

        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)

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
        y = self.classifier(f)
        return y, f

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.model.layer0(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0, x1), 1)
        y, f = self.attention(x, b, t)

        if not self.training:
            return f
        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50RNN(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50RNN, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.hidden_dim = 512
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        output, (h_n, c_n) = self.lstm(x)
        output = output.permute(0, 2, 1)
        f = F.avg_pool1d(output, t)
        f = f.view(b, self.hidden_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class Temporalpooling(nn.Module):
    def __init__(self, Feat_dim, num_classes, droprate, w, h, **kwargs):
        super(Temporalpooling, self).__init__()
        self.Feat_dim = Feat_dim
        self.middle_dim = 512
        self.attention_conv = nn.Conv2d(self.Feat_dim, self.middle_dim,
                                        [w, h])  # 7,4 cooresponds to 224, 112 input image size

        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
        self.att_gen = 'softmax'


    def forward(self, x, b, t):

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
        a = a.expand(b, t, self.Feat_dim)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)

        f = att_x.view(b, self.Feat_dim)


        return  f

class global_branch(nn.Module):
    def __init__(self,num_classes, **kwargs):
        super(global_branch, self).__init__()
        model_ft = torchvision.models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.feature_dim=2048
        self.attention = Temporalpooling(self.feature_dim, num_classes, droprate=0.5, w=1, h=1)
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self,x,b,t):
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x,y = self.attention(x, b, t)
        return x,y

class local_branch(nn.Module):
    def __init__(self,num_classes,part, **kwargs):
        super(local_branch, self).__init__()
        model_ft = torchvision.models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        self.part=part
        self.feat_dim=2048
        self.attention = Temporalpooling(self.feat_dim, num_classes, droprate=0.5, w=1, h=1)
        self.avgpool_part = nn.AdaptiveAvgPool2d((self.part, 1))
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(self.feat_dim,num_classes,droprate=0.5,relu=False,bnorm=True,num_bottleneck=256))

    def forward(self,x,b=8,t=4):
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        yg,fg = self.attention(x, b, t)

        if self.part<1:
            return yg,fg

        f=[]
        y=[]

        for i in range(self.part):
            part = torch.unsqueeze(x[:, :, i], 3)
            f.append(self.attention(part, b, t))
            name = 'classifier' + str(i)
            c = getattr(self, name)
            y.append(c(f[i]))
            return yg,fg,y,f


class TST(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, stride=2, droprate=0.5, **kwargs):
        super(TST, self).__init__()

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

        res_g_conv5 = modelft.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(modelft.layer4.state_dict())

        modelft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool =modelft.avgpool

        self.gbranch = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.pbranch = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.feature_dim = 4096
        self.attention = Temporalpooling(self.feature_dim, num_classes, droprate=0.5, w=1, h=1)


    def global_branch_fn(self,x):
        x = self.gbranch(x)
        x = self.avgpool(x)
        return x
    def part_branch(self,x):
        x = self.pbranch(x)
        x = self.avgpool(x)
        return x




    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x=self.backbone(x)
        #x=torch.cat((self.global_branch_fn(x),self.part_branch(x)),1)
        x=self.global_branch_fn(x)

        f = self.attention(x, b, t)
        if not self.training:
            return f


        if self.loss == {'xent'}:
            return f
        elif self.loss == {'xent', 'htri'}:
            return f
        elif self.loss == {'cent'}:
            return  f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))








