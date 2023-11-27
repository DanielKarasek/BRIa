import os

import torch
from torch import optim
from torch.nn import MSELoss

from dataloader import create_dataloader
from enums import NoiseTypeEnum
from models.model_factory import ModelFactory
from visualiser import visualise_gt_noised_and_predicted


def create_model(cls_cnt: int, regressinout_out_dim: int) -> torch.nn.Module:
    model = ModelFactory.AERegression(class_cnt=cls_cnt,
                                      regression_out_dim=regressinout_out_dim)
    model = model.cuda()
    return model


def create_result_dir(root_dir: str):
    file_base_name = "run"
    target_dir = "run"
    i = 0
    while target_dir in os.listdir(root_dir):
        i += 1
        target_dir = f"{file_base_name}_{i}"
    os.mkdir(f"{root_dir}/{target_dir}")
    os.mkdir(f"{root_dir}/{target_dir}/model")
    os.mkdir(f"{root_dir}/{target_dir}/plots")
    return f"{root_dir}/{target_dir}"


def train_single_epoch(model: torch.nn.Module,
                       criterion: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler,
                       dl_train: torch.utils.data.DataLoader,
                       loss_history: dict,
                       epoch: int) -> dict:
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target_regression, _) in enumerate(dl_train):
        data = data.cuda()

        target_regression = target_regression.cuda()
        _, output_regression = model(data)
        loss = criterion(output_regression, target_regression)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    running_loss /= len(dl_train.dataset)
    if epoch == 30:
        optimizer.param_groups[0]['initial_lr'] /= 10
    print(f'Train Epoch: {epoch}\n'
          f'\tMean Regression Loss: {running_loss:.6f}')

    scheduler.step()
    return loss_history


def evaluate_single_epoch(model: torch.nn.Module,
                          criterion: torch.nn.Module,
                          dl_test: torch.utils.data.DataLoader,
                          loss_history: dict) -> dict:
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for data, target_regression, _ in dl_test:
            data = data.cuda()
            target_regression = target_regression.cuda()
            _, output_regression = model(data)
            loss = criterion(output_regression, target_regression)

            running_loss += loss

    running_loss /= len(dl_test.dataset)
    print(f'\nTest set: \n'
          f'\tAverage regression loss: {running_loss:.4f}\n')
    return loss_history


def train(model: torch.nn.Module,
          criterion: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          dl_train: torch.utils.data.DataLoader,
          dl_test: torch.utils.data.DataLoader,
          root_dir: str,
          epoch_cnt: int):
    save_dir = create_result_dir(root_dir)
    loss_history = {"train": {"regression": [], "classification": []},
                    "test": {"regression": [], "classification": []}}
    for epoch in range(epoch_cnt):
        loss_history = train_single_epoch(model, criterion, optimizer, scheduler, dl_train, loss_history, epoch)
        loss_history = evaluate_single_epoch(model, criterion, dl_test, loss_history)
    torch.save(model.state_dict(), f"{save_dir}/model/model.pth")


def main():
    CLS_CNT = 3
    REGRESSION_OUT_DIM = 256
    BATCH_SIZE = 64
    EPOCH_CNT = 450

    model = create_model(cls_cnt=CLS_CNT,
                         regressinout_out_dim=REGRESSION_OUT_DIM)
    criterion = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000008)
    dl_train, dl_test = create_dataloader(BATCH_SIZE, [NoiseTypeEnum.FACIAL_MUSCLES_MOVEMENT], return_fft=True)
    # for batch_idx, (data, target_regression, target_classification) in enumerate(dl_train):
    # 30+60+120+240 = 450
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                               T_0=30,
                                                               T_mult=2,
                                                               eta_min=0.000_000_01)

    train(model, criterion, optimizer, scheduler, dl_train, dl_test, "./results", EPOCH_CNT)


if __name__ == '__main__':
    main()
