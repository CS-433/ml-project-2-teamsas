import os
from typing import List, Union

import torch
import torch.nn as nn

from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers import BigBirdTokenizer, BigBirdModel


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LanguageModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class BertLanguageModel(LanguageModel):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        max_length: int,
        aggregation: str,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.max_length = max_length
        self.aggregation = aggregation
        self.device = device

    def tokenize(self, texts: Union[str, List[str]]) -> List[List[int]]:
        return self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
        )

    def forward(self, inputs) -> torch.Tensor:
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():  # TODO
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            if self.aggregation == "mean":
                mask = (
                    inputs["attention_mask"]
                    .unsqueeze(-1)
                    .expand(last_hidden_state.size())
                )
                embeddings = (last_hidden_state * mask).sum(1) / mask.sum(1)
            elif self.aggregation == "cls":
                embeddings = last_hidden_state[:, 0, :]
        return embeddings

    def to_device(self, device: torch.device) -> "BertLanguageModel":
        self.model = self.model.to(device)
        self.device = device
        return self


class BigBirdRobertaBase(BertLanguageModel):
    def __init__(
        self,
        aggregation: str,
        device: torch.device,
    ) -> None:
        super().__init__(
            tokenizer=BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base"),
            model=BigBirdModel.from_pretrained("google/bigbird-roberta-base"),
            max_length=4096,
            aggregation=aggregation,
            device=device,
        )
        self.output_dim = 768


class FullyConnectedArchitecture(torch.nn.Module):
    def __init__(
        self,
        num_features_in: int,
        hidden_layers_size: List[int],
        non_linearity: nn.Module,
    ) -> None:
        super().__init__()
        self.num_features_in = num_features_in
        self.num_features_out = hidden_layers_size[-1]
        self.non_linearity = non_linearity
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


class MultiRegressionArchitecture(torch.nn.Module):
    def __init__(
        self,
        num_features_in: int,
        hidden_layers_size: List[int],
        num_features_out: int,
        non_linearity: nn.Module,
    ) -> None:
        super().__init__()
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        self.non_linearity = non_linearity
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

    def to_device(self, device: torch.device) -> "MultiRegressionArchitecture":
        for i, model in enumerate(self.models):
            self.models[i] = model.to(device)
        return self.to(device)


class JointRegressionArchitecture(torch.nn.Module):
    def __init__(
        self,
        num_features_in: int,
        hidden_layers_size: List[int],
        num_features_out: int,
        non_linearity: nn.Module,
    ) -> None:
        super().__init__()
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        self.non_linearity = non_linearity
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

    def to_device(self, device: torch.device) -> "JointRegressionArchitecture":
        self.encoder.to(device)
        self.nl = self.nl.to(device)
        self.ll = self.ll.to(device)
        self.model = self.model.to(device)
        return self.to(device)


class MultiRegressionWithJointEncoderArchitecture(torch.nn.Module):
    def __init__(
        self,
        num_features_in: int,
        shared_hidden_layers_size: List[int],
        independent_hidden_layers_size: List[int],
        num_features_out: int,
        non_linearity: nn.Module,
    ) -> None:
        super().__init__()
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        self.non_linearity = non_linearity
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

    def to_device(
        self, device: torch.device
    ) -> "MultiRegressionWithJointEncoderArchitecture":
        self.shared_encoder.to(device)
        self.nl = self.nl.to(device)
        for ll in self.lls:
            ll.to(device)
        return self.to(device)


class TextOnlyModel(torch.nn.Module):
    def __init__(
        self,
        embedding_generation: LanguageModel,
        regression_model: torch.nn.Module,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.embedding_generation = embedding_generation
        self.regression_model = regression_model
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.embedding_generation(x).detach()
        return self.regression_model(x)

    def to_device(self, device: torch.device) -> "TextOnlyModel":
        self.embedding_generation.to(device)
        self.regression_model.to(device)
        self.device = device
        return self.to(device)
