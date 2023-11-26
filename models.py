import torch
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
                 pretrained=True) -> nn.Module:
        return ResnetAny(class_cnt=class_cnt,
                         regression_out_dim=regression_out_dim,
                         pretrained=pretrained,
                         resnet_size=18)

    @staticmethod
    def Resnet34(class_cnt: int,
                 regression_out_dim: int,
                 pretrained=True) -> nn.Module:
        return ResnetAny(class_cnt=class_cnt,
                         regression_out_dim=regression_out_dim,
                         pretrained=pretrained,
                         resnet_size=34)

    @staticmethod
    def Resnet50(class_cnt: int,
                 regression_out_dim: int,
                 pretrained=True) -> nn.Module:
        return ResnetAny(class_cnt=class_cnt,
                         regression_out_dim=regression_out_dim,
                         pretrained=pretrained,
                         resnet_size=50)

    @staticmethod
    def Resnet101(class_cnt: int,
                  regression_out_dim: int,
                  pretrained=True) -> nn.Module:
        return ResnetAny(class_cnt=class_cnt,
                         regression_out_dim=regression_out_dim,
                         pretrained=pretrained,
                         resnet_size=101)

    @staticmethod
    def TransformerBRIaProModelPlus(class_cnt: int,
                                    regression_out_dim: int) -> nn.Module:
        return TransformerBRIaProModelPlus(class_cnt=class_cnt,
                                           regression_out_dim=regression_out_dim)

    @staticmethod
    def AE(class_cnt: int,
           regression_out_dim: int) -> nn.Module:
        return AE(class_cnt=class_cnt,
                  regression_out_dim=regression_out_dim)


class HeadClassification(nn.Module):
    def __init__(self, num_ftrs, class_cnt):
        super(HeadClassification, self).__init__()
        self.class_head_fn1 = nn.Linear(num_ftrs, class_cnt)
        # self.class_head_out = nn.Softmax()
        self._init()

    def forward(self, x):
        cls_x = self.class_head_fn1(x)
        # cls_out = self.class_head_out(cls_x)
        return cls_x

    def _init(self):
        init.kaiming_normal_(self.class_head_fn1.weight, mode='fan_out')
        init.constant_(self.class_head_fn1.bias, 0)


class HeadRegression(nn.Module):
    def __init__(self, num_ftrs, regression_out_dim):
        super(HeadRegression, self).__init__()
        self.regression_fn1 = nn.Linear(num_ftrs, regression_out_dim * 2)
        self.regression_fn1_act = nn.ReLU()
        self.regression_out = nn.Linear(regression_out_dim * 2, num_ftrs)
        self._init()

    def forward(self, x):
        reg_x = self.regression_fn1(x)
        reg_x = self.regression_fn1_act(reg_x)
        reg_out = self.regression_out(reg_x)
        return reg_out

    def _init(self):
        init.kaiming_normal_(self.regression_fn1.weight, mode='fan_out')
        init.constant_(self.regression_fn1.bias, 0)
        init.kaiming_normal_(self.regression_out.weight, mode='fan_out')
        init.constant_(self.regression_out.bias, 0)


class TransformerBRIaProModelPlus(nn.Module):
    def __init__(self,
                 class_cnt: int,
                 regression_out_dim: int):
        super(TransformerBRIaProModelPlus, self).__init__()
        d_model = 512
        self.transformer = nn.Transformer(d_model=d_model)
        self.classification_head = HeadClassification(d_model, class_cnt)
        self.regression_head = HeadRegression(d_model, regression_out_dim)

    def forward(self, x):
        x = self.transformer(x)
        cls_out = self.classification_head(x)
        reg_out = self.regression_head(x)
        return cls_out, reg_out


class AE(torch.nn.Module):
    def __init__(self, class_cnt: int, regression_out_dim: int):
        super().__init__()

        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512)
        )
        self.classification_head = HeadClassification(512, class_cnt)
        self.regression_head = HeadRegression(512, regression_out_dim)

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        cls_out = self.classification_head(decoded)
        reg_out = self.regression_head(decoded)
        return cls_out, reg_out


class ResnetAny(nn.Module):
    def __init__(self,
                 class_cnt: int,
                 regression_out_dim: int,
                 pretrained=True,
                 resnet_size: int = 18):
        super(ResnetAny, self).__init__()
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

        self.model.classification_head = HeadClassification(self.num_ftrs, self._class_cnt)
        self.model.regression_head = HeadRegression(self.num_ftrs, regression_out_dim)

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

        cls_out = self.model.classification_head(x)
        reg_out = self.model.regression_head(x)
        return cls_out, reg_out
