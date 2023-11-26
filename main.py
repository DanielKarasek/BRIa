import os

import numpy as np
import torch
from torch import optim

from dataloader import create_dataloader
from losses import Criterion
from models import ResnetFactory
import sklearn.metrics

from visualiser import create_confusion_plot, plot_losses, visualise_gt_noised_and_predicted


def create_model(cls_cnt: int, regressinout_out_dim: int) -> torch.nn.Module:
    model = ResnetFactory.AE(class_cnt=cls_cnt,
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
    for batch_idx, (data, target_regression, target_classification) in enumerate(dl_train):
        data = data.cuda()

        target_regression = target_regression.cuda()
        target_classification = target_classification.cuda()
        output_classification, output_regression = model(data)
        loss, loss_regression, loss_classification = criterion(output_regression,
                                                               output_classification,
                                                               target_regression,
                                                               target_classification)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tRegression Loss: {:.6f}\tClassification Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dl_train.dataset),
                       100. * batch_idx / len(dl_train), loss_regression.item(), loss_classification.item()))
        loss_history["train"]["regression"].append(loss_regression.item())
        loss_history["train"]["classification"].append(loss_classification.item())
    scheduler.step()
    return loss_history


def evaluate_single_epoch(model: torch.nn.Module,
                          criterion: torch.nn.Module,
                          dl_test: torch.utils.data.DataLoader,
                          loss_history: dict,
                          epoch: int,
                          save_dir: str) -> dict:
    model.eval()
    loss_classification = 0
    loss_regression = 0
    correct = 0
    all_preds = np.array([])
    all_targets = np.array([])
    with torch.no_grad():
        for data, target_regression, target_classification in dl_test:
            data = data.cuda()

            target_regression = target_regression.cuda()
            target_classification = target_classification.cuda()
            output_classification, output_regression = model(data)
            _, loss_regr, loss_cls = criterion(output_regression,
                                               output_classification,
                                               target_regression,
                                               target_classification)

            loss_classification += loss_cls
            loss_regression += loss_regr

            predictions = output_classification.argmax(dim=1, keepdim=True).detach()

            predictions: np.ndarray = predictions.cpu().numpy().squeeze()
            target_classification: np.ndarray = target_classification.cpu().numpy().squeeze()

            all_preds = np.concatenate([all_preds, predictions])
            all_targets = np.concatenate([all_targets, target_classification])
    if epoch % 50 == 0:
        create_confusion_plot(all_preds,
                              all_targets,
                              show=False,
                              save_fig=f'{save_dir}/plots/confusion_matrix_{epoch}.png')

    loss_classification /= len(dl_test.dataset)
    loss_regression /= len(dl_test.dataset)
    print('\nTest set: \n'
          '            Average classification loss: {:.4f}\n'
          '            Average regression loss: {:.4f}\n'
          '            Accuracy: {}/{} ({:.0f}%)\n'.format(loss_classification,
                                                           loss_regression,
                                                           correct,
                                                           len(dl_test.dataset),
                                                           100. * correct / len(dl_test.dataset)))
    loss_history["test"]["regression"].append(loss_regression.item())
    loss_history["test"]["classification"].append(loss_classification.item())
    return loss_history


def create_stats(model: torch.nn.Module,
                 dl_test: torch.utils.data.DataLoader,
                 save_dir: str):
    model.eval()
    all_preds = np.array([])
    all_targets = np.array([])
    with torch.no_grad():
        for data, target_regression, target_classification in dl_test:
            data = data.cuda()
            target_regression = target_regression.cuda()
            target_classification = target_classification.cuda()
            output_classification, output_regression = model(data)

            predictions = output_classification.argmax(dim=1, keepdim=True).detach()

            predictions: np.ndarray = predictions.cpu().numpy().squeeze()
            target_classification: np.ndarray = target_classification.cpu().numpy().squeeze()

            all_preds = np.concatenate([all_preds, predictions])
            all_targets = np.concatenate([all_targets, target_classification])

    f1_score = sklearn.metrics.f1_score(all_targets, all_preds, average='macro')
    recall_score = sklearn.metrics.recall_score(all_targets, all_preds, average='macro')
    precision_score = sklearn.metrics.precision_score(all_targets, all_preds, average='macro')
    accuracy_score = sklearn.metrics.accuracy_score(all_targets, all_preds)
    # save scores to file
    with open(f"{save_dir}/scores.txt", "w") as f:
        f.write(f"F1 score: {f1_score}\n")
        f.write(f"Recall score: {recall_score}\n")
        f.write(f"Precision score: {precision_score}\n")
        f.write(f"Accuracy score: {accuracy_score}\n")
    create_confusion_plot(all_preds,
                          all_targets,
                          show=False,
                          save_fig=f'{save_dir}/plots/confusion_matrix_final.png')


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
        loss_history = evaluate_single_epoch(model, criterion, dl_test, loss_history, epoch, save_dir)
    plot_losses(loss_history, save_dir=f"{save_dir}/plots/regression_train.png")
    create_stats(model, dl_test, save_dir)
    torch.save(model.state_dict(), f"{save_dir}/model/model.pth")


def main():
    CLS_CNT = 3
    REGRESSION_OUT_DIM = 512
    BATCH_SIZE = 64
    EPOCH_CNT = 450

    model = create_model(cls_cnt=CLS_CNT,
                         regressinout_out_dim=REGRESSION_OUT_DIM)
    criterion = Criterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    dl_train, dl_test = create_dataloader(BATCH_SIZE)
    # for batch_idx, (data, target_regression, target_classification) in enumerate(dl_train):
    #     dirty = data[0]
    #     clean = target_regression[0]
    #     cls = target_classification[0]
    #     visualise_gt_noised_and_predicted(dirty, clean, clean)
    # 30+60+120+240 = 450
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                               T_0=30,
                                                               T_mult=2,
                                                               eta_min=0.0000001)

    train(model, criterion, optimizer, scheduler, dl_train, dl_test, "./results", EPOCH_CNT)


if __name__ == '__main__':
    main()
