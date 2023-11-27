from torch import nn as nn

from models.AE import AE
from models.transformer import Transformer


class ModelFactory:
    @staticmethod
    def AE(class_cnt: int,
           regression_out_dim: int) -> nn.Module:
        return AE(class_cnt=class_cnt,
                  regression_out_dim=regression_out_dim)

    @staticmethod
    def tranformer_classification(class_cnt: int,
                                  regression_out_dim: int) -> nn.Module:
        return Transformer(class_cnt=class_cnt,
                           regression_out_dim=regression_out_dim)
