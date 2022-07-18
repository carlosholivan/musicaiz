from pathlib import Path
from typing import Union, Dict

from musicaiz.datasets.configs import MusicGenerationDataset
from musicaiz.datasets.utils import tokenize_path
from musicaiz.tokenizers import MMMTokenizer


class LakhMIDI(MusicGenerationDataset):
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
    
    @staticmethod
    def get_metadata(
        dataset_path: str,
        train_split: float = 0.7,
        test_split: float = 0.2
    ) -> Dict[str, str]:
        """

        Args:
            dataset_path (str): _description_
            train_split (float): _description_
            test_split (float): _description_

            validation split is automatically calculated as:
            1 - train_split - test_split

        Returns:
            _type_: _description_
        """

        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)

        composers_json = {}
        # iterate over subdirs which are different artists
        for composer_path in dataset_path.glob("*/"):
            # 1. Process composer
            composer = composer_path.stem
            # Some composers are written with 2 different composers separated by "/"
            # we'll only consider the 1st one
            composer = composer.replace(" ", "-")
            composer = composer.upper()

            # iterate over songs of an artist
            songs = [f for f in composer_path.glob("*/")]
            n_songs = len(songs)

            train_idxs = int(round(n_songs * train_split))
            val_idxs = int(n_songs * test_split)

            train_seqs = songs[:train_idxs]
            val_seqs = songs[train_idxs:val_idxs+train_idxs]
            test_seqs = songs[val_idxs+train_idxs:]


            # split in train, validation and test
            # we do this here to ensure that every artist is  at least in
            # the training and test sets (if n_songs > 1)
            for song in songs:
                if song in train_seqs:
                    split = "train"
                elif song in val_seqs:
                    split = "validation"
                else:
                    split = "test"
        
                composers_json.update(
                    {
                        composer_path.stem + "/" + song.name: {
                            "composer": composer,
                            "split": split
                        }
                    }
                )
        return composers_json


def _tokenize_multiproc(
    dataset_path: str,
    output_path: str,
    output_file: str
):

    metadata = LakhMIDI.get_metadata(dataset_path)

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


