from torch import nn as nn
from torch.nn import init as init


class ClassificationHead(nn.Module):
    def __init__(self, num_ftrs, class_cnt):
        super(ClassificationHead, self).__init__()
        self.class_head_fn1 = nn.Linear(num_ftrs, class_cnt)
        self._init()

    def forward(self, x):
        cls_x = self.class_head_fn1(x)
        return cls_x

    def _init(self):
        init.kaiming_normal_(self.class_head_fn1.weight, mode='fan_out')
        init.constant_(self.class_head_fn1.bias, 0)
