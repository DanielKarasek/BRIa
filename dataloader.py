import torch
from dataset import DatasetEEGNoise


def create_dataloader(batch_size):
    ds = DatasetEEGNoise("clean.npy", "eyes.npy", "muscles.npy", 300)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4
    )

    ds = DatasetEEGNoise("clean.npy", "eyes.npy", "muscles.npy")
    val_dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4
    )

    return dl, val_dl
