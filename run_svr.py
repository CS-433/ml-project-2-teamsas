from data_loader import (
    get_inputs_my_personality,
    get_inputs_data,
    get_inputs_chunked_data,
)
from learning import Regression_SVR
import argparse
from pathlib import Path


def run_svr(
    type, datapath_features, datapath_targets, target, features, datapath_features2=None
):
    if type == "my_personality":
        X_my_personality, y_my_personality = get_inputs_my_personality(
            datapath_features=datapath_features,
            datapath_targets=datapath_targets,
            features=features,
            datapath_features2=datapath_features2,
        )
        Regression_SVR(
            X=X_my_personality,
            y=y_my_personality,
            type="my_personality",
            target=target,
            features=features,
        )

    elif type == "idiap":
        X_idiap, y_idiap = get_inputs_data(
            datapath_features=datapath_features,
            datapath_targets=datapath_targets,
            features=features,
            datapath_features2=datapath_features2,
        )
        Regression_SVR(
            X=X_idiap, y=y_idiap, type="idiap", target=target, features=features
        )

    elif type == "idiap_chunked":
        X_idiap, y_idiap = get_inputs_chunked_data(
            datapath_features=datapath_features,
            datapath_targets=datapath_targets,
            features=features,
            datapath_features2=datapath_features2,
        )
        Regression_SVR(
            X=X_idiap, y=y_idiap, type="idiap_chunked", target=target, features=features
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature Extraction")

    parser.add_argument(
        "--type",
        type=str,
        help="Type of data to be used",
        choices=["my_personality", "idiap", "idiap_chunked"],
    )
    parser.add_argument(
        "--datapath_features",
        type=Path,
        help="Path to the features",
    )
    parser.add_argument(
        "--datapath_targets",
        type=Path,
        help="Path to the targets",
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Target to be predicted",
    )
    parser.add_argument(
        "--features",
        type=str,
        help="Features to be used",
    )
    parser.add_argument(
        "--datapath_features2",
        type=Path,
        help="Path to the features",
        default=None,
    )

    args = parser.parse_args()

    run_svr(
        type=args.type,
        datapath_features=args.datapath_features,
        datapath_targets=args.datapath_targets,
        target=args.target,
        features=args.features,
        datapath_features2=args.datapath_features2,
    )


if __name__ == "__main__":
    main()
