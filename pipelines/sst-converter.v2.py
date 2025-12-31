from argparse import ArgumentParser
from pathlib import Path

import whisper

import logging

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

MODEL_SIZE = "tiny"

# ==========================================================================================


def extract_audio(input_path: Path) -> Path: ...


def transcribe_audio(input_path: Path, output_path: Path):
    model = whisper.load_model(MODEL_SIZE)
    result = model.transcribe(
        audio=str(input_path),
    )
    logging.debug(f"result: {result=}")

    text = result.get("text").strip()

    with open(output_path, "w") as f:
        f.write(text)

    return text


def main(input: str, output: str, **kwargs):
    logging.debug(f"main: {args=}, {kwargs=}")

    if not (input_path := Path(input)).resolve().exists():
        raise FileNotFoundError(input_path)

    if output_path := Path(output).resolve():
        try:
            output_path.touch()
        except Exception as err:
            raise Exception(f"Cannot create output file: {output_path}") from err

    res = transcribe_audio(input_path, output_path)
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

    if args.get("load_model"):
        logging.debug(f"Loaded model: WIP")

    main(**args)
