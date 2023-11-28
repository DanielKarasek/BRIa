from torch import nn as nn

from models.AE import AE_class, AERegression
from models.NNet import NNet
from models.transformer import Transformer


class ModelFactory:
    @staticmethod
    def AERegression(class_cnt: int, regression_out_dim: int) -> nn.Module:
        return AERegression(class_cnt=class_cnt, regression_out_dim=regression_out_dim)

    @staticmethod
    def tranformer_classification(class_cnt: int, regression_out_dim: int) -> nn.Module:
        return Transformer(class_cnt=class_cnt, regression_out_dim=regression_out_dim)

    @staticmethod
    def AE_class(class_cnt: int, regression_out_dim: int) -> nn.Module:
        return AE_class(class_cnt=class_cnt, regression_out_dim=regression_out_dim)

    @staticmethod
    def NNet(class_cnt: int, regression_out_dim: int) -> nn.Module:
        return NNet(class_cnt=class_cnt, regression_out_dim=regression_out_dim)
