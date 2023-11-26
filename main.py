import torch
from torch import optim

from losses import Criterion
from models import ResnetFactory


def get_dataloaders(batch_size: int) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    return None, None


def create_model(cls_cnt: int, regressinout_out_dim: int) -> torch.nn.Module:
    model = ResnetFactory.Resnet18(class_cnt=cls_cnt,
                                   regression_out_dim=regressinout_out_dim,
                                   pretrained=True)
    model.cuda()
    return model


def train(model: torch.nn.Module,
          criterion: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          dl_train: torch.utils.data.DataLoader,
          dl_test: torch.utils.data.DataLoader,
          epoch_cnt: int):
    for epoch in range(epoch_cnt):
        # train
        model.train()
        for batch_idx, (data, target_regression, target_classification) in enumerate(dl_train):
            output_regression, output_classification = model(data)
            loss, loss_classification, loss_regression = criterion(output_regression,
                                                                   output_classification,
                                                                   target_regression,
                                                                   target_classification)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tRegression Loss: {:.6f}\tClassification Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(dl_train.dataset),
                           100. * batch_idx / len(dl_train), loss_regression.item(), loss_classification.item()))
        scheduler.step()
        # test
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target_regression, target_classification in dl_test:
                output_regression, output_classification = model(data)
                test_loss += criterion(output_regression, output_classification, target_regression,
                                       target_classification).item()
                pred = output_classification.data.max(1, keepdim=True)[1]
                correct += pred.eq(target_classification.data.view_as(pred)).cpu().sum()

        test_loss /= len(dl_test.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(dl_test.dataset),
            100. * correct / len(dl_test.dataset)))
def main():
    CLS_CNT = 3
    REGRESSION_OUT_DIM = 512
    BATCH_SIZE = 64
    EPOCH_CNT = 450

    model = create_model(cls_cnt=CLS_CNT,
                         regressinout_out_dim=REGRESSION_OUT_DIM)
    criterion = Criterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    dl_train, dl_test = get_dataloaders(BATCH_SIZE)
    # 30+60+120+240 = 210
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                               T_0=30,
                                                               T_mult=2,
                                                               eta_min=0.0000001)

    train(model, criterion, optimizer, scheduler, dl_train, dl_test, EPOCH_CNT)
