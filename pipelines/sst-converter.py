from argparse import ArgumentParser

from pathlib import Path

import logging

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# ==========================================================================================


def main(*args, **kwargs):
    logging.debug(f"main: {args=}, {kwargs=}")
    pass


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="SSTConverter",
        description="Speech-to-Text converter with langchain and OpenAI interface.",
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to input audio files: mp3, wav.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        default="output.txt",
        help="Path to output text file, as raw text file.",
    )

    args = vars(parser.parse_args())

    main(**args)

# ==========================================================================================
