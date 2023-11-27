from torch import nn

from models.backends.transformer_1d import Transformer1d
from models.heads.classification_head import ClassificationHead
from models.heads.regression_head import RegressionHead


class Transformer(nn.Module):
    def __init__(self, class_cnt: int, regression_out_dim: int):
        super(Transformer, self).__init__()
        self.backend = Transformer1d()
        self.classification_head = ClassificationHead(512, class_cnt)
        self.regression_head = RegressionHead(512, regression_out_dim)

    def forward(self, x):
        x = self.backend(x)
        cls_out = self.classification_head(x)
        reg_out = self.regression_head(x)
        return cls_out, reg_out
