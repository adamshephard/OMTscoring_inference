import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Function

from models.net_utils import Identitiy

class MLP(nn.Module):
    def __init__(self, d, hidden_d, nr_classes):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(d, hidden_d)
        self.out = nn.Linear(hidden_d, nr_classes)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = x.float()
        x = x.view(x.size(0), -1)
        x = self.hidden1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.out(x)
        # x = torch.sigmoid(x)
        return x

class FC(nn.Module):
    def __init__(self, d, nr_classes):
        super(FC, self).__init__()
        self.out = nn.Linear(d, nr_classes)
    def forward(self, x):
        x = x.float()
        x = x.view(x.size(0), -1)
        x = self.out(x)
        # x = torch.sigmoid(x)
        return x

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class ResNet34_DA(nn.Module):
    def __init__(self, nr_classes, nr_domains):
        super(ResNet34_DA, self).__init__()
        self.model = models.resnet34(True) #pretrained resnet34
        num_ftrs = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_ftrs, nr_classes)
        self.model.fc = Identity()
        self.class_classifier = nn.Linear(num_ftrs, nr_classes) # potentially add extra layers here
        self.domain_classifier = nn.Linear(num_ftrs, nr_domains) # potentially add extra layers here
    def forward(self, input_data, alpha):
        feature = self.model(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        # domain_output = self.domain_classifier(feature)
        return class_output, domain_output

