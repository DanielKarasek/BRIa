import torch

from models.heads.classification_head import ClassificationHead
from models.heads.regression_head import RegressionHead


class AERegression(torch.nn.Module):
    def __init__(self, class_cnt: int, regression_out_dim: int):
        super().__init__()

        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Identity()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Identity()
        )
        self.classification_head = ClassificationHead(512, class_cnt)
        self.regression_head = RegressionHead(512, regression_out_dim)

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        cls_out = self.classification_head(decoded)
        reg_out = self.regression_head(decoded)
        return cls_out, reg_out


class AE_class(torch.nn.Module):
    def __init__(self, class_cnt: int, regression_out_dim: int):
        super().__init__()

        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512)
        )
        self.classification_head = ClassificationHead(512, class_cnt)
        self.regression_head = RegressionHead(512, regression_out_dim)

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        cls_out = self.classification_head(decoded)
        reg_out = self.regression_head(decoded)
        return cls_out, reg_out