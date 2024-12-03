import argparse
from pathlib import Path

import pandas as pd

from src import data


LANGS = [
    "french",
]

TARGET_CORPUS = [
    "reuters",
    "brown",
]


def main() -> None:
    parser = argparse.ArgumentParser()

    # NOTE: general parameters
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="path to the input data. expected format is csv/xlsx.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="path to the output data.",
    )
    parser.add_argument(
        "-n",
        "--n_aug",
        type=int,
        required=False,
        default=1,
        help="number of augmented data.",
    )
    parser.add_argument(
        "--row_start",
        type=int,
        required=False,
        default=0,
        help="first (inclusive) row number.",
    )
    parser.add_argument(
        "--row_end",
        type=int,
        required=False,
        default=-1,
        help="last (exclusive) row numbers.",
    )

    # NOTE: parameters for translation method
    parser.add_argument(
        "--translation",
        action="store_true",
        help="activate translation augmentation method.",
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        required=False,
        choices=LANGS,
        default="french",
        help="target language for translation.",
    )

    # NOTE: parameters for noise_injection method
    parser.add_argument(
        "--noise_injection",
        action="store_true",
        help="activate noise injection augmentation method.",
    )
    parser.add_argument(
        "--char_insert_aug_p",
        type=float,
        required=False,
        default=0.0,
        help="probability of char insert augmentation.",
    )
    parser.add_argument(
        "--ocr_aug_p",
        type=float,
        required=False,
        default=0.0,
        help="probability of ocr augmentation.",
    )
    parser.add_argument(
        "--word_swapping_aug_p",
        type=float,
        required=False,
        default=0.0,
        help="probability of word swapping augmentation.",
    )

    # NOTE: parameters for tf_idf_based method
    parser.add_argument(
        "--tf_idf_dropping_p",
        type=float,
        required=False,
        default=0.0,
        help="probability of tf-idf dropping augmentation.",
    )
    parser.add_argument(
        "--tf_idf_syn_replace_p",
        type=float,
        required=False,
        default=0.0,
        help="probability of replacing with tf_idf synonym.",
    )
    parser.add_argument(
        "--target_corpus",
        type=str,
        choices=TARGET_CORPUS,
        required=False,
        help="target corpus for tf-idf dropping augmentation.",
    )

    # NOTE: BERT masking based augmentation
    parser.add_argument(
        "--masking_p",
        type=float,
        required=False,
        default=0.0,
        help="probability of masking augmentation.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="bert-base-uncased",
        help="name of the model for masking.",
    )

    args = parser.parse_args()

    input_path = args.input
    assert input_path.exists(), "input file does not exist."
    assert input_path.is_file(), "input path should be a file."
    assert input_path.suffix in [".csv", ".xlsx"], "input file should be csv or xlsx."

    if input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix == ".xlsx":
        df = pd.read_excel(input_path)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_aug = args.n_aug
    assert n_aug > 0, "n_aug should be greater than 0."

    row_start = args.row_start
    row_end = args.row_end
    assert row_start >= 0, "row_start should be greater than or equal to 0."
    assert row_end >= -1, "row_end should be greater than or equal to -1."
    assert row_start < row_end, "row_start should be less than row_end."
    if not (row_start == 0 and row_end == -1):
        assert row_end <= len(df), "row_end should be less than the number of rows."
        df = df.iloc[row_start:row_end]

    translation = args.translation
    source_lang = args.source_lang
    target_lang = args.target_lang
    if translation:
        assert target_lang is not None, "target lang should be provided."
        assert source_lang != target_lang, "different source and target langs."

    noise_injection = args.noise_injection
    char_insert_aug = args.char_insert_aug
    ocr_aug = args.ocr_aug
    word_swapping_aug = args.word_swapping_aug
    word_deleting_aug = args.word_deleting_aug
    if noise_injection:
        assert (
            0 <= char_insert_aug <= 1
            and 0 <= ocr_aug <= 1
            and 0 <= word_swapping_aug <= 1
            and 0 <= word_deleting_aug <= 1
        ), "char_insert_aug, ocr_aug, word_swapping_aug, word_deleting_aug should be between 0 and 1."
        assert (
            char_insert_aug + ocr_aug + word_swapping_aug + word_deleting_aug > 0
        ), "at least one augmentation method should be activated."
    else:
        assert (
            char_insert_aug == 0
            and ocr_aug == 0
            and word_swapping_aug == 0
            and word_deleting_aug == 0
        ), "inconsistent parameters, noise_injection is not activated while char_insert_aug, ocr_aug, word_swapping_aug, word_deleting_aug are not zeros."

    tf_idf_based = args.tf_idf_based
    tf_idf_dropping_p = args.tf_idf_dropping_p
    tf_idf_syn_replace_p = args.tf_idf_syn_replace_p
    target_corpus = args.target_corpus
    if tf_idf_based:
        assert (
            0 <= tf_idf_dropping_p <= 1 and 0 <= tf_idf_syn_replace_p <= 1
        ), "tf_idf_dropping_p, tf_idf_syn_replace_p should be between 0 and 1."
        assert (
            tf_idf_dropping_p + tf_idf_syn_replace_p > 0
        ), "at least one augmentation method should be activated."
        assert target_corpus is not None, "target corpus should be provided."
    else:
        assert (
            tf_idf_dropping_p == 0 and tf_idf_syn_replace_p == 0
        ), "inconsistent parameters, tf_idf_based is not activated while tf_idf_dropping_p, tf_idf_syn_replace_p are not zeros."

    masking_p = args.masking_p
    model_name = args.model_name
    assert 0 <= masking_p <= 1, "masking_p should be between 0 and 1."

    # TODO: call the augmentation and cleaning methods


if __name__ == "__main__":
    main()
