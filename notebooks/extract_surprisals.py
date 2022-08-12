# %%
from collections import Counter
from functools import partial
import argparse

from matplotlib import pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
import surprisal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        required=True,
        help="huggingface model id to use for extracting surprisals",
    )
    parser.add_argument(
        "-c",
        "--model_class",
        type=str,
        required=False,
        default=None,
        choices=["bert", "gpt"],
        help="huggingface model type",
    )
    parser.add_argument("--output_dir", type=str, default=".", help="output directory")

    args = parser.parse_args()

    model = surprisal.AutoHuggingFaceModel.from_pretrained(
        args.model_name_or_path, model_class=args.model_class
    )

    # %%
    df = pd.read_csv("vecchi2016_an_data_cogsci/annotations.csv")[
        ["unit_id", "which_makes_more_sense", "an1", "an2"]
    ]
    df.head(4)

    # %%
    all_pairs = df.an1.to_list() + df.an2.to_list()
    all_df = pd.DataFrame({"an": list(set(all_pairs))})
    all_df

    # %%
    surprisals = [
        *map(
            partial(model.extract_surprisal, prefix="How likely is this: ", suffix=""),
            tqdm(all_df.an.iloc[:]),
        )
    ]

    all_df[args.model_name_or_path] = surprisals
    all_df.to_csv(
        f"{args.output_dir}/vecchi2016_an_surprisals_{args.model_name_or_path}.csv"
    )


if __name__ == "__main__":
    main()
