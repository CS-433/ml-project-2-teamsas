from typing import List

import torch
import torch.nn as nn


class LanguageModel(torch.nn.Module):
    # TODO: for finetune
    def tokenize(self, text: List[str]) -> List[List[int]]:
        pass

    def forward(self) -> torch.Tensor:
        pass


class Architecture(torch.nn.Module):
    def __init__(
        self,
        num_features_in: int,
        num_features_out: int,
        non_linearity: nn.Module,
    ) -> None:
        super().__init__()
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        self.non_linearity = non_linearity


class FullyConnectedArchitecture(Architecture):
    def __init__(
        self,
        num_features_in: int,
        hidden_layers_size: List[int],
        non_linearity: nn.Module,
    ) -> None:
        super().__init__(num_features_in, hidden_layers_size[-1], non_linearity)
        self.hidden_layers_size = hidden_layers_size
        layers_sizes = zip(
            [num_features_in] + hidden_layers_size[:-1],
            hidden_layers_size,
        )
        self.layers = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(layers_sizes):
            self.layers.append(nn.Linear(in_size, out_size))
            if i != len(hidden_layers_size) - 1:
                self.layers.append(non_linearity)
        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiRegressionArchitecture(Architecture):
    def __init__(
        self,
        num_features_in: int,
        hidden_layers_size: List[int],
        num_features_out: int,
        non_linearity: nn.Module,
    ) -> None:
        super().__init__(num_features_in, num_features_out, non_linearity)
        self.hidden_layers_size = hidden_layers_size
        self.models = []
        for _ in range(num_features_out):
            self.encoder = FullyConnectedArchitecture(
                num_features_in,
                hidden_layers_size,
                non_linearity,
            )
            self.nl = non_linearity
            self.ll = nn.Linear(hidden_layers_size[-1], 1)
            model = nn.Sequential(self.encoder, self.nl, self.ll)
            self.models.append(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([model(x) for model in self.models], dim=1)


class JointRegressionArchitecture(Architecture):
    def __init__(
        self,
        num_features_in: int,
        hidden_layers_size: List[int],
        num_features_out: int,
        non_linearity: nn.Module,
    ) -> None:
        super().__init__(num_features_in, num_features_out, non_linearity)
        self.hidden_layers_size = hidden_layers_size
        self.encoder = FullyConnectedArchitecture(
            num_features_in,
            hidden_layers_size,
            non_linearity,
        )
        self.nl = non_linearity
        self.ll = nn.Linear(hidden_layers_size[-1], num_features_out)
        self.model = nn.Sequential(self.encoder, self.nl, self.ll)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiRegressionWithJointEncoderArchitecture(Architecture):
    def __init__(
        self,
        num_features_in: int,
        shared_hidden_layers_size: List[int],
        independent_hidden_layers_size: List[int],
        num_features_out: int,
        non_linearity: nn.Module,
    ) -> None:
        super().__init__(num_features_in, num_features_out, non_linearity)
        self.shared_hidden_layers_size = shared_hidden_layers_size
        self.joint_embedding_size = shared_hidden_layers_size[-1]
        self.independent_hidden_layers_size = independent_hidden_layers_size
        self.shared_encoder = FullyConnectedArchitecture(
            num_features_in,
            shared_hidden_layers_size,
            non_linearity,
        )
        self.nl = non_linearity
        self.lls = [
            FullyConnectedArchitecture(
                self.joint_embedding_size,
                independent_hidden_layers_size + [1],
                non_linearity,
            )
            for _ in range(num_features_out)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nl(self.shared_encoder(x))
        x = torch.cat([ll(x) for ll in self.lls], dim=1)
        return x


class TextOnlyModel(object):
    def __init__(
        self,
        embedding_generation: LanguageModel,
        regression_model: Architecture,
    ) -> None:
        self.embedding_generation = embedding_generation
        self.regression_model = regression_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regression_model(self.embedding_generation(x))
