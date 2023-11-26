import torch.nn as nn
import torch.nn.init as init
from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import resnet101


class ResnetFactory:
    @staticmethod
    def Resnet18(class_cnt: int,
                 regression_out_dim: int,
                 pretrained=True, ):
        return ResnetAny(class_cnt=class_cnt,
                         regression_out_dim=regression_out_dim,
                         pretrained=pretrained,
                         resnet_size=18)

    @staticmethod
    def Resnet34(class_cnt: int,
                 regression_out_dim: int,
                 pretrained=True, ):
        return ResnetAny(class_cnt=class_cnt,
                         regression_out_dim=regression_out_dim,
                         pretrained=pretrained,
                         resnet_size=34)

    @staticmethod
    def Resnet50(class_cnt: int,
                 regression_out_dim: int,
                 pretrained=True, ):
        return ResnetAny(class_cnt=class_cnt,
                         regression_out_dim=regression_out_dim,
                         pretrained=pretrained,
                         resnet_size=50)

    @staticmethod
    def Resnet101(class_cnt: int,
                  regression_out_dim: int,
                  pretrained=True, ):
        return ResnetAny(class_cnt=class_cnt,
                         regression_out_dim=regression_out_dim,
                         pretrained=pretrained,
                         resnet_size=101)


class ResnetAny:
    def __init__(self,
                 class_cnt: int,
                 regression_out_dim: int,
                 pretrained=True,
                 resnet_size: int = 18):
        assert resnet_size == 18 or resnet_size == 34 or resnet_size == 50 or resnet_size == 101
        if resnet_size == 18:
            self.model = resnet18(pretrained)
        elif resnet_size == 34:
            self.model = resnet34(pretrained)
        elif resnet_size == 50:
            self.model = resnet50(pretrained)
        elif resnet_size == 101:
            self.model = resnet101(pretrained)
        self._class_cnt = class_cnt
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.dropout = nn.Dropout()
        self.model.class_head_fn1 = nn.Linear(self.num_ftrs, self._class_cnt)
        self.model.class_head_out = nn.Softmax(self._class_cnt)
        self.model.regression_fn1 = nn.Linear(self.num_ftrs, regression_out_dim * 2)
        self.model.regression_fn1_act = nn.ReLU()
        self.model.regression_out = nn.Linear(regression_out_dim * 2, self.num_ftrs)

        self._initialize_weights()

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x

        x = x.view(x.size(0), -1)

        x = self.model.dropout(x)

        cls_x = self.model.class_head_fn1(x)
        cls_out = self.model.class_head_out(cls_x)

        reg_x = self.model.regression_fn1(x)
        reg_x = self.model.regression_fn1_act(reg_x)
        reg_out = self.model.regression_out(reg_x)

        return cls_out, reg_out

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.class_head_fn1.weight, mode='fan_out')
        init.constant_(self.model.class_head_fn1.bias, 0)
        init.kaiming_normal_(self.model.regression_fn1.weight, mode='fan_out')
        init.constant_(self.model.regression_fn1.bias, 0)
        init.kaiming_normal_(self.model.regression_out.weight, mode='fan_out')
        init.constant_(self.model.regression_out.bias, 0)
