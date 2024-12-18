import argparse
from feature_extraction import (
    creating_my_personality_data,
    creating_idiap_data,
    creating_idiap_chunked_data,
    saving_NRC_data,
    saving_NRC_VAD_data,
    saving_LIWC_data,
    saving_readability_data,
    Saving_aggregated_features,
)
from pathlib import Path


def feature_extraction(
    type,
    datapath,
    NRC_path,
    output_path_NRC,
    NRC_VAD_PATH,
    output_path_NRC_VAD,
    output_path_readability,
    output_path_LIWC,
    output_path,
    targret_path,
):

    if type == "my_personality":
        X_my_personality, y_my_personality = creating_my_personality_data(
            datapath=datapath
        )
        saving_NRC_data(
            NRC_path=NRC_path,
            X=X_my_personality,
            output_path=output_path_NRC,
            type="my_personality",
        )
        saving_NRC_VAD_data(
            NRC_path=NRC_VAD_PATH,
            X=X_my_personality,
            output_path=output_path_NRC_VAD,
            type="my_personality",
        )
        saving_readability_data(
            X=X_my_personality,
            output_path=output_path_readability,
            type="my_personality",
        )
        saving_LIWC_data(
            X=X_my_personality, output_path=output_path_LIWC, type="my_personality"
        )
        Saving_aggregated_features(
            y=y_my_personality,
            X=X_my_personality,
            type="my_personality",
            Path_NRC=output_path_NRC,
            Path_NRC_VAD=output_path_NRC_VAD,
            Path_readability=output_path_readability,
            Path_LIWC=output_path_LIWC,
            output_path=output_path,
            target_path=targret_path,
        )

    elif type == "idiap":
        X_idiap, y_idiap, data_idiap = creating_idiap_data(datapath=datapath)
        saving_NRC_data(
            NRC_path=NRC_path, X=X_idiap, output_path=output_path_NRC, type="idiap"
        )
        saving_NRC_VAD_data(
            NRC_path=NRC_VAD_PATH,
            X=X_idiap,
            output_path=output_path_NRC_VAD,
            type="idiap",
        )
        saving_readability_data(
            X=X_idiap, output_path=output_path_readability, type="idiap"
        )
        saving_LIWC_data(X=X_idiap, output_path=output_path_LIWC, type="idiap")
        Saving_aggregated_features(
            y=y_idiap,
            X=X_idiap,
            type="idiap",
            Path_NRC=output_path_NRC,
            Path_NRC_VAD=output_path_NRC_VAD,
            Path_readability=output_path_readability,
            Path_LIWC=output_path_LIWC,
            output_path=output_path,
            target_path=targret_path,
        )

    elif type == "idiap_chunked":
        X_idiap_chunked, y_idiap_chunked, data_idiap_chunked = (
            creating_idiap_chunked_data(datapath=datapath)
        )
        saving_NRC_data(
            NRC_path=NRC_path,
            X=X_idiap_chunked,
            output_path=output_path_NRC,
            type="idiap_chunked",
        )
        saving_NRC_VAD_data(
            NRC_path=NRC_VAD_PATH,
            X=X_idiap_chunked,
            output_path=output_path_NRC_VAD,
            type="idiap_chunked",
        )
        saving_readability_data(
            X=X_idiap_chunked, output_path=output_path_readability, type="idiap_chunked"
        )
        saving_LIWC_data(
            X=X_idiap_chunked, output_path=output_path_LIWC, type="idiap_chunked"
        )
        Saving_aggregated_features(
            y=y_idiap_chunked,
            X=X_idiap_chunked,
            type="idiap_chunked",
            Path_NRC=output_path_NRC,
            Path_NRC_VAD=output_path_NRC_VAD,
            Path_readability=output_path_readability,
            Path_LIWC=output_path_LIWC,
            output_path=output_path,
            target_path=targret_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature Extraction")

    parser.add_argument(
        "--type",
        type=str,
        help="Type of data to be used for feature extraction",
        required=True,
    )
    parser.add_argument(
        "--datapath",
        type=Path,
        help="Path to the data",
        required=True,
    )
    parser.add_argument(
        "--NRC_path",
        type=Path,
        help="Path to the NRC lexicon",
        required=True,
        default="./lib_/NRC-Emotion-Lexicon.xlsx",
    )
    parser.add_argument(
        "--output_path_NRC",
        type=Path,
        help="Path to save the NRC features",
        required=True,
    )
    parser.add_argument(
        "--NRC_VAD_PATH",
        type=Path,
        help="Path to the NRC VAD lexicon",
        required=True,
        default="./lib_/NRC-VAD-Lexicon.txt",
    )
    parser.add_argument(
        "--output_path_NRC_VAD",
        type=Path,
        help="Path to save the NRC VAD features",
        required=True,
    )
    parser.add_argument(
        "--output_path_readability",
        type=Path,
        help="Path to save the readability features",
        required=True,
    )
    parser.add_argument(
        "--output_path_LIWC",
        type=Path,
        help="Path to save the LIWC features",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to save the aggregated features",
        required=True,
    )
    parser.add_argument(
        "--target_path",
        type=Path,
        help="Path to save the target",
        required=True,
    )

    args = parser.parse_args()

    feature_extraction(
        type=args.type,
        datapath=args.datapath,
        NRC_path=args.NRC_path,
        output_path_NRC=args.output_path_NRC,
        NRC_VAD_PATH=args.NRC_VAD_PATH,
        output_path_NRC_VAD=args.output_path_NRC_VAD,
        output_path_readability=args.output_path_readability,
        output_path_LIWC=args.output_path_LIWC,
        output_path=args.output_path,
        targret_path=args.target_path,
    )


if __name__ == "__main__":
    main()
