from pathlib import Path
from typing import Union
import pandas as pd

from musicaiz.datasets.configs import MusicGenerationDataset
from musicaiz.datasets.utils import tokenize_path
from musicaiz.tokenizers import MMMTokenizer


class MAESTRO(MusicGenerationDataset):
    """
    """

    def __init__(self, dir: str):
        self.dir = dir

    @staticmethod
    def tokenize(
        dataset_path: Union[Path, str],
        output_path: Union[Path, str],
        tokenizer: str = "MMM",
        output_filename: str = "token-sequences",
    ):
        if tokenizer == "MMM":
            _tokenize_multiproc(
                dataset_path=dataset_path,
                output_file=output_filename,
                output_path=output_path
            )
            _ = MMMTokenizer.get_vocabulary(
                dataset_path=output_path
            )


def get_metadata(table: pd.DataFrame):

    composers_json = {}
    for index, row in table.iterrows():
        # 1. Process composer
        composer = row["canonical_composer"]
        # Some composers are written with 2 different composers separated by "/"
        # we'll only consider the 1st one
        composer = composer.split("/")[0]
        composers_json.update(
            {
                row["midi_filename"]: {
                    "split": row["split"]
                }
            }
        )
    return composers_json


def _tokenize_multiproc(
    dataset_path: str,
    output_path: str,
    output_file: str = "token-sequences",
    csv_file: str = "maestro-v2.0.0.csv"
):

    dataset_path = str(Path(dataset_path, "maestro-v2.0.0"))
    table = pd.read_csv(str(Path(dataset_path, csv_file)))
    metadata = get_metadata(table)
    # Split metadata in train, validation and test
    train_metadata, val_metadata, test_metadata = {}, {}, {}
    for key, val in metadata.items():
        if val["split"] == "train":
            train_metadata.update({key: val})
        elif val["split"] == "validation":
            val_metadata.update({key: val})
        elif val["split"] == "test":
            test_metadata.update({key: val})
        else:
            continue

    # Midis are distributes as
    data_path = Path(dataset_path)

    # make same dirs to store the token sequences separated in
    # train, valid and test
    dest_train_path = Path(output_path, "train")
    dest_train_path.mkdir(parents=True, exist_ok=True)

    dest_val_path = Path(output_path, "validation")
    dest_val_path.mkdir(parents=True, exist_ok=True)

    dest_test_path = Path(output_path, "test")
    dest_test_path.mkdir(parents=True, exist_ok=True)

    tokenize_path(data_path, dest_train_path, train_metadata, output_file)
    tokenize_path(data_path, dest_val_path, val_metadata, output_file)
    tokenize_path(data_path, dest_test_path, test_metadata, output_file)
