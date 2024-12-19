import numpy as np
from keras.utils import pad_sequences
import torch


def pad_text(texts, tokenizer, max_len):
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=max_len)


def train_epoch(model, criterion, optimizer, train_loader, device):
    model.train()
    train_loss = 0.0
    total_samples = 0

    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(inputs)
        total_samples += len(inputs)

    return train_loss / total_samples


def validate_epoch(model, criterion, val_loader, device):
    """
    Validate the model and compute MAE and RMSE for each output variable.
    """
    model.eval()
    val_loss = 0.0
    val_targets_all = []
    val_outputs_all = []
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * len(inputs)
            total_samples += len(inputs)
            val_targets_all.append(targets.cpu().numpy())
            val_outputs_all.append(outputs.cpu().detach().numpy())

    val_loss /= total_samples
    val_targets_all = np.concatenate(val_targets_all)
    val_outputs_all = np.concatenate(val_outputs_all)

    # Per-output MAE and RMSE
    per_output_mae = np.mean(np.abs(val_outputs_all - val_targets_all), axis=0)
    per_output_rmse = np.sqrt(np.mean((val_outputs_all - val_targets_all) ** 2, axis=0))

    # Overall MAE and RMSE (averaged across all outputs)
    overall_mae = np.mean(per_output_mae)
    overall_rmse = np.mean(per_output_rmse)

    return val_loss, overall_mae, overall_rmse, per_output_mae, per_output_rmse
