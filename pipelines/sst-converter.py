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

    support_group = parser.add_argument_group("support")
    support_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG level for logging.",
    )
    support_group.add_argument(
        "--load-model",
        action="store_true",
        help="Load model into memory.",
    )

    args = vars(parser.parse_args())

    if args.get("debug") and logging.getLogger().setLevel(logging.DEBUG):
        logging.debug("DEBUG logging enabled")

    main(**args)

# ==========================================================================================
