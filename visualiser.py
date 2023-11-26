import matplotlib.pyplot as plt
import numpy as np


def visualise_gt_noised_and_predicted(noised, gt, predicted):
    plt.figure(figsize=(20, 10))
    plt.plot(noised, label='noised')
    plt.plot(gt, label='gt')
    plt.plot(predicted, label='predicted')
    plt.legend()
    plt.show()


def main():
    true = np.load('data/clean.npy')
    noise = np.load('data/muscles.npy')
    noised_example = true[0] + noise[0]*0.008
    visualise_gt_noised_and_predicted(noised_example, true[0], true[0])


if __name__ == '__main__':
    main()