import torch.nn


class Criterion(torch.nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self._regression_crit = torch.nn.MSELoss()
        self._classification_crit = torch.nn.CrossEntropyLoss()

    def forward(self, output_regression, output_classification, target_regression, target_classification):
        loss_regression = self._regression_crit(output_regression, target_regression)
        loss_classification = self._classification_crit(output_classification, target_classification)
        return loss_regression + loss_classification, loss_regression, loss_classification


