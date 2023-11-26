from typing import List

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import seaborn as sns
import pandas as pd

def visualise_gt_noised_and_predicted(noised, gt, predicted):
    plt.figure(figsize=(20, 10))
    plt.plot(noised, label='noised')
    plt.plot(gt, label='gt')
    plt.plot(predicted, label='predicted')
    plt.legend()
    plt.show()


def create_confusion_plot(predicted: np.ndarray,
                          target: np.ndarray,
                          show: bool = False,
                          save_fig: str = ''):
    confusion_matrix = sklearn.metrics.confusion_matrix(target, predicted)
    classes = ["clean", "EOG noise", "EMG noise"]
    df_cm = pd.DataFrame(confusion_matrix / np.sum(confusion_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sns.heatmap(df_cm, annot=True)
    if save_fig:
        plt.savefig('output.png')
    if show:
        plt.show()


def plot_losses(loss_history: dict, save_dir: str = ''):
    plot_loss(loss_history["train"]["regression"], "Train regression loss", ["step", "value"], save_dir=save_dir)
    plot_loss(loss_history["train"]["classification"], "Train classification loss", ["step", "value"], save_dir=save_dir)
    plot_loss(loss_history["test"]["regression"], "Test regression loss", ["epoch", "value"], save_dir=save_dir)
    plot_loss(loss_history["test"]["classification"], "Test classification loss", ["epoch", "value"], save_dir=save_dir)


def plot_loss(data: List[float],
              title: str,
              axis_names: List[str] = ["epoch", "loss"],
              save_dir: str = ''):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    if save_dir:
        plt.savefig(save_dir)

def main():
    true = np.load('data/clean.npy')
    noise = np.load('data/muscles.npy')
    noised_example = true[0] + noise[0]*0.008
    visualise_gt_noised_and_predicted(noised_example, true[0], true[0])


if __name__ == '__main__':
    main()