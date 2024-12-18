import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import accuracy_score


# NOTE: These are the thresholds which has been used to discretize the targets for the MyPersonality dataset in the literature.
# NOTE: We have extracted these thresholds from the dataset in notebook `thresholds_of_my_personality.ipynb`.
DEFAULT_MY_PERSONALITY_THRESHOLDS = np.array(
    [
        0.611716621253406,
        0.43571428571428567,
        0.5597014925373134,
        0.5704225352112676,
        0.5545454545454545,
    ]
)


def get_threshold_for_target(target_name: str) -> np.array:
    """
    Get the threshold for a specific target.

    Parameters
    ----------
    target_name : str
        The name of the target variable.

    Returns
    -------
    np.array
        The threshold for the target variable.
    """
    if target_name == "sEXT":
        thr = DEFAULT_MY_PERSONALITY_THRESHOLDS[0]
    elif target_name == "sNEU":
        thr = DEFAULT_MY_PERSONALITY_THRESHOLDS[1]
    elif target_name == "sAGR":
        thr = DEFAULT_MY_PERSONALITY_THRESHOLDS[2]
    elif target_name == "sCON":
        thr = DEFAULT_MY_PERSONALITY_THRESHOLDS[3]
    elif target_name == "sOPN":
        thr = DEFAULT_MY_PERSONALITY_THRESHOLDS[4]
    else:
        raise ValueError(f"Invalid target name: {target_name}")
    return np.array([thr])


def measure_performance(
    y_true: np.array,
    y_pred: np.array,
    use_classification_metrics: bool = False,
    thresholds_for_classification: np.array = None,
):
    """
    Measure the performance of a model using a variety of metrics.

    Parameters
    ----------
    y_true : np.array
        The true values of the target variables. Shape is (n_samples, n_targets).

    y_pred : np.array
        The predicted values of the target variables. Shape is (n_samples, n_targets).

    use_classification_metrics : bool
        Whether to use classification metrics. If False, only regression metrics are used.

    thresholds_for_classification : np.array
        The thresholds to use for classification metrics. If None, the default thresholds are used.
    """
    assert y_true.shape == y_pred.shape

    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    mae_per_target = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    rmse_per_target = root_mean_squared_error(y_true, y_pred, multioutput="raw_values")
    mean_mae = np.mean(mae_per_target)
    mean_rmse = np.mean(rmse_per_target)
    if use_classification_metrics:
        if thresholds_for_classification is None:
            thresholds_for_classification = DEFAULT_MY_PERSONALITY_THRESHOLDS

        assert y_true.shape[1] == len(thresholds_for_classification)
        accuracy_per_target = []
        for i in range(y_true.shape[1]):
            y_true_class = y_true[:, i] > thresholds_for_classification[i]
            y_pred_class = y_pred[:, i] > thresholds_for_classification[i]
            accuracy = accuracy_score(y_true_class, y_pred_class)
            accuracy_per_target.append(accuracy)
        accuracy_per_target = np.array(accuracy_per_target)
        mean_accuracy = np.mean(accuracy_per_target)
        return {
            "mean_mae": mean_mae,
            "mean_rmse": mean_rmse,
            "mean_accuracy": mean_accuracy,
            "mae_per_target": mae_per_target,
            "rmse_per_target": rmse_per_target,
            "accuracy_per_target": accuracy_per_target,
        }
    else:
        return {
            "mean_mae": mean_mae,
            "mean_rmse": mean_rmse,
            "mae_per_target": mae_per_target,
            "rmse_per_target": rmse_per_target,
        }
