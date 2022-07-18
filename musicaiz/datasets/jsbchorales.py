from pathlib import Path
from typing import List, Union

from musicaiz.datasets.configs import MusicGenerationDataset
from musicaiz.datasets.utils import tokenize_path
from musicaiz.tokenizers import MMMTokenizer


class JSBChorales(MusicGenerationDataset):
    """The JSB Chorales dataset is organized in 3 directories:
        - train
        - valid
        - test
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


def _tokenize_multiproc(dataset_path: str, output_file: str, output_path: str):
    # Midis are distributes as
    data_train_path = Path(dataset_path, "train")
    data_val_path = Path(dataset_path, "valid")
    data_test_path = Path(dataset_path, "test")

    # make same dirs to store the token sequences separated in
    # train, valid and test
    dest_train_path = Path(output_path, "train")
    dest_train_path.mkdir(parents=True, exist_ok=True)

    dest_val_path = Path(output_path, "validation")
    dest_val_path.mkdir(parents=True, exist_ok=True)

    dest_test_path = Path(output_path, "test")
    dest_test_path.mkdir(parents=True, exist_ok=True)

    tokenize_path(data_train_path, dest_train_path, None, output_file)
    tokenize_path(data_val_path, dest_val_path, None, output_file)
    tokenize_path(data_test_path, dest_test_path, None, output_file)
