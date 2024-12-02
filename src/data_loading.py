from typing import List, Optional, Tuple

import torch
import torch.utils.data


class PersonalityDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        target_vars: torch.Tensor,
        texts: List[List[str]],  # TODO: tokenized texts
    ) -> None:
        assert len(target_vars) == len(texts), "inconsistent lengths."
        self.num_samples = len(target_vars)
        self.target_vars = target_vars
        self.texts = texts
        self.normalization_params = None

    def normalize_targets(
        self,
        params: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:
        if params is None:
            mean = torch.mean(self.target_vars, dim=0)
            std = torch.std(self.target_vars, dim=0)
        else:
            mean, std = params
        self.target_vars = (self.target_vars - mean) / std
        self.normalization_params = mean, std
        return mean, std

    def denormalize_targets(self, predictions: torch.Tensor) -> torch.Tensor:
        mean, std = self.normalization_params
        return predictions * std + mean

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.side_features is not None:
            return self.texts[idx], self.side_features[idx], self.target_vars[idx]
        return self.texts[idx], self.target_vars[idx]


def train_test_split(
    dataset: PersonalityDataset, ratio: float, shuffle: bool = True
) -> Tuple[PersonalityDataset, PersonalityDataset]:
    num_samples = len(dataset)
    indices = list(range(num_samples))
    if shuffle:
        torch.random.shuffle(indices)
    split_idx = int(num_samples * ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    return train_dataset, test_dataset
