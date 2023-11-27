import torch
from torch import nn as nn


from models.heads.classification_head import ClassificationHead
from models.heads.regression_head import RegressionHead


class NNet(torch.nn.Module):
    # inspired by https://www.kaggle.com/code/banggiangle/cnn-eeg-pytorch
    def __init__(self, class_cnt: int, regression_out_dim: int):
        super(NNet, self).__init__()

        self.hidden = 32
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=self.hidden, kernel_size=5, padding=0
            ),
            nn.LeakyReLU(0.1),
            nn.Conv1d(
                in_channels=self.hidden,
                out_channels=self.hidden,
                kernel_size=5,
                padding=0,
            ),
            nn.LeakyReLU(0.1),
            # nn.MaxPool1d(2, 2),
            # nn.Dropout(p=0.1, inplace=True),
            nn.Conv1d(self.hidden, 1, 1),
        )
        self.classification_head = ClassificationHead(248, class_cnt)
        self.regression_head = RegressionHead(248, regression_out_dim)

    def forward(self, x: torch.Tensor):
        # print(f"x old: {x.shape}")
        x = x[:, None, :]
        # print(f"x new: {x.shape}")
        y = self.net(x)
        # print(f"y old: {y.shape}")
        y = y[:, 0, :]
        # print(f"y new: {y.shape}")

        cls_out = self.classification_head(y)
        reg_out = self.regression_head(y)
        return cls_out, reg_out
