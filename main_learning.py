import argparse
import math
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import data_loading
from src import learning
from src import models


MODELS = [
    "multi-reg",
    "joint-reg",
    "multi-reg-joint-emb",
]

BACKBONES = [
    "big-bird-roberta-base",
]

NON_LINS = [
    "relu",
    "tanh",
    "sigmoid",
    "leaky_relu",
    "elu",
    "gelu",
]

CRITERIONS = [
    "mae",
    "mse",
]

OPTIMIZERS = [
    "adam",
]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        required=True,
        help="path to the dataset. expected format is csv.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="path to the configuration file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="path to the output directory.",
    )

    args = parser.parse_args()

    input_path = args.data
    assert input_path.exists(), f"{input_path} does not exist."
    assert input_path.is_file(), f"{input_path} is not a file."
    assert input_path.suffix in [".csv", ".xlsx"], f"{input_path} is not a csv file."

    config_path = args.config
    assert config_path.exists(), f"{config_path} does not exist."
    assert config_path.is_file(), f"{config_path} is not a file."
    assert config_path.suffix in [".yml", ".yaml"], f"{config_path} is not a yaml file."

    output_path = args.output
    output_path.mkdir(parents=True, exist_ok=True)
    assert output_path.exists(), f"{output_path} does not exist."
    assert output_path.is_dir(), f"{output_path} is not a directory."

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    target_vars_names = config["target_vars"]
    assert isinstance(target_vars_names, list), "target_vars should be a list."
    assert len(target_vars_names) > 0, "target_vars should not be empty."

    model_config = config["model"]
    backbone_name = model_config["backbone"]
    backbone_aggregation = model_config["backbone_aggregation"]
    assert backbone_name in BACKBONES, f"{backbone_name} is not a valid backbone."
    model_name = model_config["name"]
    assert model_name in MODELS, f"{model_name} is not a valid model."
    non_lin_name = model_config["arch"]["non_linearity"]
    assert non_lin_name in NON_LINS, f"{non_lin_name} is not a valid non-linearity."

    validation_split = config["training"]["validation_split"]
    assert 0 < validation_split < 1, "validation_split should be in (0, 1)."

    num_epochs = config["training"]["num_epochs"]
    assert num_epochs > 0, "num_epochs should be positive."

    batch_size = config["training"]["batch_size"]
    assert batch_size > 0, "batch_size should be positive."

    criterion_name = config["training"]["criterion"]
    assert criterion_name in CRITERIONS, f"{criterion_name} is not a valid criterion."
    optimizer_name = config["training"]["optimizer"]["name"]
    assert optimizer_name in OPTIMIZERS, f"{optimizer_name} is not a valid optimizer."

    early_stopping_monitor = config["training"]["early_stopping"]["monitor"]
    early_stopping_patience = config["training"]["early_stopping"]["patience"]

    checkpoint_frequency = config["training"]["checkpoint_frequency"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if backbone_name == "big-bird-roberta-base":
        backbone = models.BigBirdRobertaBase(
            aggregation=backbone_aggregation,
            device=device,
        )
        lm_output_dim = backbone.output_dim
    else:
        KeyError(f"{backbone_name} is not a valid backbone.")

    if non_lin_name == "relu":
        non_lin = torch.nn.ReLU()
    elif non_lin_name == "tanh":
        non_lin = torch.nn.Tanh()
    elif non_lin_name == "sigmoid":
        non_lin = torch.nn.Sigmoid()
    elif non_lin_name == "leaky_relu":
        non_lin = torch.nn.LeakyReLU()
    elif non_lin_name == "elu":
        non_lin = torch.nn.ELU()
    elif non_lin_name == "gelu":
        non_lin = torch.nn.GELU()
    else:
        KeyError(f"{non_lin_name} is not a valid non-linearity.")

    if model_name == "multi-reg":
        hidden_layers_size = model_config["arch"]["hidden_layers"]
        assert isinstance(hidden_layers_size, list), "hidden_layers should be a list."
        reg_model = models.MultiRegressionArchitecture(
            num_features_in=lm_output_dim,
            hidden_layers_size=hidden_layers_size,
            num_features_out=len(target_vars_names),
            non_linearity=non_lin,
        )
    elif model_name == "joint-reg":
        hidden_layers_size = model_config["arch"]["hidden_layers"]
        assert isinstance(hidden_layers_size, list), "hidden_layers should be a list."
        reg_model = models.JointRegressionArchitecture(
            num_features_in=lm_output_dim,
            hidden_layers_size=hidden_layers_size,
            num_features_out=len(target_vars_names),
            non_linearity=non_lin,
        )
    elif model_name == "multi-reg-joint-emb":
        shared_layers_size = model_config["arch"]["shared_layers"]
        assert isinstance(shared_layers_size, list), "shared_layers should be a list."
        independent_layers_size = model_config["arch"]["independent_layers"]
        assert isinstance(
            independent_layers_size, list
        ), "independent_layers should be a list."
        reg_model = models.MultiRegressionWithJointEncoderArchitecture(
            num_features_in=lm_output_dim,
            shared_hidden_layers_size=shared_layers_size,
            independent_hidden_layers_size=independent_layers_size,
            non_linearity=non_lin,
        )
    else:
        KeyError(f"{model_name} is not a valid model.")

    backbone = backbone.to(device)
    reg_model = reg_model.to(device)
    
    end_to_end_model = models.TextOnlyModel(
        embedding_generation=backbone,
        regression_model=reg_model,
        device=device,
    )

    if criterion_name == "mse":
        criterion = torch.nn.MSELoss()
    elif criterion_name == "mae":
        criterion = torch.nn.L1Loss()
    else:
        KeyError(f"{criterion_name} is not a valid criterion.")

    if optimizer_name == "adam":
        learning_rate = config["training"]["optimizer"]["learning_rate"]
        optimizer = torch.optim.Adam(
            end_to_end_model.parameters(),
            lr=learning_rate,
        )
    else:
        KeyError(f"{optimizer_name} is not a valid optimizer.")

    if input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix == ".xlsx":
        df = pd.read_excel(input_path)

    target_vars = df[target_vars_names].to_numpy()
    target_vars = torch.tensor(target_vars, dtype=torch.float)

    print("target vars shape:", target_vars.shape)

    all_inputs = {}
    texts = df["final_text"].tolist()
    for i in tqdm(range(math.ceil(len(texts) / batch_size))):
        batch_texts = texts[i * batch_size : (i + 1) * batch_size]
        inputs = backbone.tokenize(batch_texts)
        # inputs = {key: value.to(device) for key, value in inputs.items()}
        if i == 0:
            for key in inputs.keys():
                all_inputs[key] = []
        for key, value in inputs.items():
            all_inputs[key].append(value)
    all_inputs = {key: torch.cat(value, dim=0) for key, value in all_inputs.items()}

    print("all inputs shape:", {key: value.shape for key, value in all_inputs.items()})

    dataset = data_loading.PersonalityDataset(
        tokenized_texts=all_inputs,
        target_vars=target_vars,
    )

    train_dataset, val_dataset = data_loading.train_test_split(dataset, ratio=0.8)
    normalization_params = train_dataset.normalize_targets()
    val_dataset.normalize_targets(normalization_params)

    print("train dataset size:", len(train_dataset))
    print("val dataset size:", len(val_dataset))

    trainer = learning.Trainer(
        model=end_to_end_model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size),
        val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=batch_size),
        num_epochs=num_epochs,
        early_stopping_monitor=early_stopping_monitor,
        early_stopping_patience=early_stopping_patience,
        best_model_path=output_path / "best_model.pt",
        checkpoint_path=output_path / "checkpoint.pt",
        checkpoint_frequency=checkpoint_frequency,
        device=device,
    )
    train_hist, val_hist = trainer.train()

    # TODO: save the model


if __name__ == "__main__":
    main()
