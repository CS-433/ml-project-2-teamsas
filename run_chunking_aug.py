import argparse
import pandas as pd
from tqdm import tqdm
from src.data_loader import create_chunks


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        required=True,
        help="Path to the input dataset file",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        required=True,
        help="Path to the output dataset file",
    )

    args = parser.parse_args()

    data_df = pd.read_excel(args.input_file)
    data_df = data_df.dropna()
    data_df = data_df.reset_index(drop=True)
    data_df = data_df.drop(columns=["Unnamed: 0"])

    all_chunks = []
    for participant_id, text in tqdm(
        zip(data_df["participant_id"], data_df["final_text"]),
        total=data_df.shape[0],
    ):
        chunks = create_chunks(text)
        all_chunks.append([participant_id, chunks])

    all_participants_ids = []
    all_chunks_ids = []
    all_chunks_texts = []

    for pair in all_chunks:
        participant_id, chunks = pair
        for chunk_id, chunk in enumerate(chunks):
            all_participants_ids.append(participant_id)
            all_chunks_ids.append(chunk_id)
            all_chunks_texts.append(chunk)

    chunked_data_df = pd.DataFrame(
        {
            "participant_id": all_participants_ids,
            "chunk_id": all_chunks_ids,
            "chunk_text": all_chunks_texts,
        }
    )

    final_df = pd.merge(data_df, chunked_data_df, on="participant_id", how="inner")
    final_df = final_df.drop(columns=["final_text"])
    final_df = final_df[final_df["chunk_text"].apply(lambda x: len(x.split()) >= 50)]
    final_df.to_csv(args.output_file)


if __name__ == "__main__":
    main()
