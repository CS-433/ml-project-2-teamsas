from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.data


class PersonalityDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenized_texts: Dict[str, torch.Tensor],
        target_vars: torch.Tensor,
    ) -> None:
        assert len(tokenized_texts) > 0, "no tokenized texts."
        for k in tokenized_texts.keys():
            assert len(tokenized_texts[k]) == len(target_vars), "inconsistent lengths."
        self.num_samples = len(target_vars)
        self.tokenized_texts = tokenized_texts
        self.target_vars = target_vars
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
        tokenized_texts_idx = {k: t[idx] for k, t in self.tokenized_texts.items()}
        return tokenized_texts_idx, self.target_vars[idx]


def train_test_split(
    dataset: PersonalityDataset, ratio: float
) -> Tuple[PersonalityDataset, PersonalityDataset]:
    num_samples = len(dataset)
    indices = torch.randperm(num_samples)
    split_idx = int(num_samples * ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    train_dataset = PersonalityDataset(
        {k: v[train_indices] for k, v in dataset.tokenized_texts.items()},
        dataset.target_vars[train_indices],
    )
    test_dataset = PersonalityDataset(
        {k: v[test_indices] for k, v in dataset.tokenized_texts.items()},
        dataset.target_vars[test_indices],
    )
    return train_dataset, test_dataset
