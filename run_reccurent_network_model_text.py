import argparse
import pickle
from src.cleaning import clean_pipe_line

import numpy as np
import pandas as pd
from keras.utils import pad_sequences
from sklearn.model_selection import KFold

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models import TextModel
from src.models_utils import pad_text, train_epoch, validate_epoch
from src.early_stopping import EarlyStopping

embedding_matrix = np.load("./fast_text/embedding.npy")

with open("./fast_text/tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)


MAX_SEQUENCE_LENGTH = 280
TEXT_COLUMN = "text"
EMBEDDINGS_DIMENSION = 300


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="my_personality",
        help="Dataset to use for training",
    )
    args = parser.parse_args()
    dataset = args.dataset

    if dataset == "my_personality":
        PERSONALITY_COLUMNS = [
            "sEXT",
            "sNEU",
            "sAGR",
            "sCON",
            "sOPN",
        ]
        train = pd.read_csv(
            "data/my_personality/my_personality.csv",
            encoding="ISO-8859-1",
        )
        train.rename(columns={"STATUS": "text"}, inplace=True)
    elif dataset == "idiap":
        PERSONALITY_COLUMNS = [
            "hones16",
            "emoti16",
            "extra16",
            "agree16",
            "consc16",
            "openn16",
            "icar_hat0",
            "icar_hat1",
            "icar_hat2",
        ]
        train = pd.read_excel("data/idiap/dataset.xlsx")
        train.rename(columns={"final_text": "text"}, inplace=True)
    else:
        PERSONALITY_COLUMNS = [
            "hones16",
            "emoti16",
            "extra16",
            "agree16",
            "consc16",
            "openn16",
            "icar_hat0",
            "icar_hat1",
            "icar_hat2",
        ]
        train = pd.read_csv("data/idiap_chunked")
        train.rename(columns={"chunk_text": "text"}, inplace=True)

    train = train.dropna()
    train = train.reset_index(drop=True)
    train = train.loc[:, PERSONALITY_COLUMNS + ["text"]]

    for col in PERSONALITY_COLUMNS:
        train[col] = train[col] - train[col].min()
        train[col] = train[col] / train[col].max()

    train_df = clean_pipe_line(train)

    def objective(trial):
        num_units = trial.suggest_int("num_units", 32, 128, step=16)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        text_data = pad_text(train_df[TEXT_COLUMN], tokenizer, MAX_SEQUENCE_LENGTH)
        labels = train_df[PERSONALITY_COLUMNS].values
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        fold_val_losses = []
        fold_val_maes = []
        fold_val_rmses = []
        fold_per_output_maes = []
        fold_per_output_rmses = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for train_idx, val_idx in kfold.split(text_data):
            train_text, val_text = text_data[train_idx], text_data[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]

            train_dataset = TensorDataset(
                torch.tensor(train_text, dtype=torch.long),
                torch.tensor(train_labels, dtype=torch.float32),
            )
            val_dataset = TensorDataset(
                torch.tensor(val_text, dtype=torch.long),
                torch.tensor(val_labels, dtype=torch.float32),
            )

            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=256)

            model = TextModel(
                vocab_size=len(tokenizer.word_index) + 1,
                embedding_dim=EMBEDDINGS_DIMENSION,
                embedding_matrix=embedding_matrix,
                num_units=num_units,
                num_layers=num_layers,
                dropout=dropout,
                num_classes=len(PERSONALITY_COLUMNS),
            )
            model = model.to(device)
            criterion = nn.L1Loss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            early_stopping = EarlyStopping(patience=3)

            for epoch in range(50):
                train_epoch(model, criterion, optimizer, train_loader, device)
                val_loss, overall_mae, overall_rmse, per_output_mae, per_output_rmse = (
                    validate_epoch(model, criterion, val_loader, device)
                )

                early_stopping(val_loss)
                if early_stopping.early_stop:
                    break

            fold_val_losses.append(val_loss)
            fold_val_maes.append(overall_mae)
            fold_val_rmses.append(overall_rmse)
            fold_per_output_maes.append(per_output_mae)
            fold_per_output_rmses.append(per_output_rmse)

        mean_val_loss = np.mean(fold_val_losses)
        std_val_loss = np.std(fold_val_losses)
        mean_val_mae = np.mean(fold_val_maes)
        std_val_mae = np.std(fold_val_maes)
        mean_val_rmse = np.mean(fold_val_rmses)
        std_val_rmse = np.std(fold_val_rmses)

        mean_per_output_mae = np.mean(fold_per_output_maes, axis=0)
        mean_per_output_rmse = np.mean(fold_per_output_rmses, axis=0)

        trial.set_user_attr("mean_val_loss", mean_val_loss)
        trial.set_user_attr("std_val_loss", std_val_loss)
        trial.set_user_attr("mean_val_mae", mean_val_mae)
        trial.set_user_attr("std_val_mae", std_val_mae)
        trial.set_user_attr("mean_val_rmse", mean_val_rmse)
        trial.set_user_attr("std_val_rmse", std_val_rmse)
        trial.set_user_attr("mean_per_output_mae", mean_per_output_mae.tolist())
        trial.set_user_attr("mean_per_output_rmse", mean_per_output_rmse.tolist())

        return mean_val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2, n_jobs=1)

    print("\n=== Best Hyperparameters ===")
    print(study.best_params)

    print("\n=== Best Validation Loss ===")
    print(study.best_value)

    best_trial = study.best_trial
    print("\n=== K-Fold Metrics for Best Trial ===")
    print(
        f"Mean Val Loss: {best_trial.user_attrs['mean_val_loss']:.4f} ± {best_trial.user_attrs['std_val_loss']:.4f}"
    )
    print(
        f"Mean Val MAE:  {best_trial.user_attrs['mean_val_mae']:.4f} ± {best_trial.user_attrs['std_val_mae']:.4f}"
    )
    print(
        f"Mean Val RMSE: {best_trial.user_attrs['mean_val_rmse']:.4f} ± {best_trial.user_attrs['std_val_rmse']:.4f}"
    )

    print("\n=== Per-Output Metrics for Best Trial ===")
    for i, column in enumerate(PERSONALITY_COLUMNS):
        print(
            f"{column}: MAE = {best_trial.user_attrs['mean_per_output_mae'][i]:.4f}, RMSE = {best_trial.user_attrs['mean_per_output_rmse'][i]:.4f}"
        )


if __name__ == "__main__":
    main()
