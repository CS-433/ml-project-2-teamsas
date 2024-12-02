import argparse
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import torch

from src import data_loading
from src import learning
from src import models


MODELS = [
    "multi-reg",
    "joint-reg",
    "multi-reg-joint-emb",
]

BACKBONES = [
    "bert",
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
    assert input_path.suffix == ".csv", f"{input_path} is not a csv file."

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

    target_vars = config["target_vars"]
    assert isinstance(target_vars, list), "target_vars should be a list."
    assert len(target_vars) > 0, "target_vars should not be empty."

    model_config = config["model"]
    backbone_name = model_config["backbone"]
    assert backbone_name in BACKBONES, f"{backbone_name} is not a valid backbone."
    model_name = model_config["model"]
    assert model_name in MODELS, f"{model_name} is not a valid model."
    non_lin_name = model_config["arch"]["non_lin"]
    assert non_lin_name in NON_LINS, f"{non_lin_name} is not a valid non-linearity."

    validation_split = model_config["training"]["validation_ratio"]
    assert 0 < validation_split < 1, "validation_ratio should be in (0, 1)."

    num_epochs = model_config["training"]["num_epochs"]
    assert num_epochs > 0, "num_epochs should be positive."

    batch_size = model_config["training"]["batch_size"]
    assert batch_size > 0, "batch_size should be positive."

    criterion_name = model_config["training"]["criterion"]
    assert criterion_name in CRITERIONS, f"{criterion_name} is not a valid criterion."
    optimizer_name = model_config["training"]["optimizer"]["name"]
    assert optimizer_name in OPTIMIZERS, f"{optimizer_name} is not a valid optimizer."

    early_stopping_monitor = model_config["training"]["early_stopping"]["monitor"]
    early_stopping_patience = model_config["training"]["early_stopping"]["patience"]

    checkpoint_frequency = model_config["training"]["checkpoint_frequency"]

    if backbone_name == "bert":
        # TODO: implement BERT model
        lm_output_dim = 768
        pass
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
            num_features_out=len(target_vars),
            non_linearity=non_lin,
        )
    elif model_name == "joint-reg":
        hidden_layers_size = model_config["arch"]["hidden_layers"]
        assert isinstance(hidden_layers_size, list), "hidden_layers should be a list."
        reg_model = models.JointRegressionArchitecture(
            num_features_in=lm_output_dim,
            hidden_layers_size=hidden_layers_size,
            num_features_out=len(target_vars),
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

    end_to_end_model = models.TextOnlyModel(
        backbone=reg_model,
        target_vars=target_vars,
    )

    if criterion_name == "mse":
        criterion = torch.nn.MSELoss()
    elif criterion_name == "mae":
        criterion = torch.nn.L1Loss()
    else:
        KeyError(f"{criterion_name} is not a valid criterion.")

    if optimizer_name == "adam":
        learning_rate = model_config["training"]["optimizer"]["learning_rate"]
        optimizer = torch.optim.Adam(lr=learning_rate)
    else:
        KeyError(f"{optimizer_name} is not a valid optimizer.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    df = pd.read_csv(input_path)

    # TODO: tokenize the text data

    # TODO: create a dataset and dataloader
    dataset = data_loading.PersonalityDataset()

    train_dataset, val_dataset = data_loading.train_test_split(dataset, ratio=0.8)

    trainer = learning.Trainer(
        model=end_to_end_model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=torch.utils.data.DataLoader(train_dataset),
        val_loader=torch.utils.data.DataLoader(val_dataset),
        num_epochs=num_epochs,
        early_stopping_monitor=early_stopping_monitor,
        early_stopping_patience=early_stopping_patience,
        checkpoint_path=output_path / "checkpoint.pt",
        checkpoint_frequency=checkpoint_frequency,
        device=device,
    )

    # TODO: save the model


if __name__ == "__main__":
    main()
