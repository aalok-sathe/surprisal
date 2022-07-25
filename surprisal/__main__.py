from argparse import ArgumentParser

import surprisal


if __name__ == "__main__":
    parser = ArgumentParser("surprisal")
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        help="name/id (on huggingface) or path to checkpoints of the language model to use for extracting surprisals",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--model_class",
        help="class of the model, must be either 'gpt' or 'bert'",
        choices=["bert", "gpt"],
        required=False,
        default=None,
    )
    parser.add_argument(
        "text",
        help="the text to score using a language model",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="whether we should plot the surprisals using matplotlib",
    )

    args = parser.parse_args()

    m = surprisal.AutoHuggingFaceModel.from_pretrained(
        args.model_name_or_path, model_class=args.model_class
    )

    [surp] = m.surprise(args.text)

    if args.plot:
        from matplotlib import pyplot as plt

        surp.lineplot()
        plt.show()

    print(surp)
