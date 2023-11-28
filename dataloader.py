import torch
from dataset import DatasetEEGNoise
from enums import NoiseTypeEnum


def create_dataloader(
    batch_size, noise_types: list[NoiseTypeEnum], return_fft: bool = False
):
    ds = DatasetEEGNoise(
        "data/clean.npy", "data/eyes.npy", "data/muscles.npy", noise_types, return_fft
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
    )

    ds = DatasetEEGNoise(
        "data/clean.npy",
        "data/eyes.npy",
        "data/muscles.npy",
        noise_types,
        return_fft,
        300,
    )
    val_dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0
    )

    return dl, val_dl
