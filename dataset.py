#!/usr/bin/env python3

import numpy as np
from torch.utils.data import Dataset
import random
from enums import NoiseTypeEnum


class DatasetEEGNoise(Dataset):
    def __init__(
        self,
        eeg_segments_path: str,
        eog_segments_path: str,
        emg_segments_path: str,
    ) -> None:
        self._eeg_segments = np.load(eeg_segments_path)
        self._eog_segments = np.load(eog_segments_path)
        self._emg_segments = np.load(emg_segments_path)

    def __len__(self) -> int:
        return len(self._eeg_segments)

    def __getitem__(self, index: int) -> tuple[int, int, NoiseTypeEnum]:
        clean_eeg_segment = self._eeg_segments[index]

        # get random noise type
        category = random.choice(list(NoiseTypeEnum))

        # combine clean and noisy segments
        match category:
            case NoiseTypeEnum.CLEAN:
                combined_eeg_segment = clean_eeg_segment
            case NoiseTypeEnum.OCULAR_ARTIFACTS:
                # get random noisy segment with ocular artifacts
                noisy_segment_index = random.randrange(0, len(self._eog_segments))
                noisy_eeg_segment = self._eog_segments[noisy_segment_index]

                # combine clean and noisy segments
                combined_eeg_segment = self._combine_clean_and_noisy_segments(
                    clean_eeg_segment, noisy_eeg_segment
                )
            case NoiseTypeEnum.MYOGENIC_ARTIFACTS:
                # get random noisy segment with myogenic artifacts
                noisy_segment_index = random.randrange(0, len(self._emg_segments))
                noisy_eeg_segment = self._emg_segments[noisy_segment_index]

                # combine clean and noisy segments
                combined_eeg_segment = self._combine_clean_and_noisy_segments(
                    clean_eeg_segment, noisy_eeg_segment
                )
            case _:
                raise ValueError(f"Unknown noise type: {category}")

        return clean_eeg_segment, combined_eeg_segment, category

    def _combine_clean_and_noisy_segments(
        self,
        clean_segments: np.ndarray,
        noisy_segments: np.ndarray,
        lambda_param: float = 0.5,
    ) -> np.ndarray:
        return clean_segments + lambda_param * noisy_segments
