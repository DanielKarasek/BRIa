import torch
from dataset import DatasetEEGNoise


def create_dataloader(batch_size):
    ds = DatasetEEGNoise("data/clean.npy",
                         "data/eyes.npy", "data/muscles.npy")
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
    )

    ds = DatasetEEGNoise("data/clean.npy", "data/eyes.npy", "data/muscles.npy", 300)
    val_dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0
    )


    return dl, val_dl
