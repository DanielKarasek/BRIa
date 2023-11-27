from torch import nn as nn
from torch.nn import init as init


class RegressionHead(nn.Module):
    def __init__(self, num_ftrs, regression_out_dim):
        super(RegressionHead, self).__init__()
        self.regression_fn1 = nn.Linear(num_ftrs, regression_out_dim // 2)
        self.regression_fn1_act = nn.ReLU()
        self.regression_out = nn.Linear(regression_out_dim // 2, regression_out_dim)
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
