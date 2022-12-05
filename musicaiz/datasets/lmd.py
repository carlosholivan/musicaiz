from pathlib import Path
from typing import Union, Dict, Type

from musicaiz.datasets.configs import (
    MusicGenerationDataset,
    MusicGenerationDatasetNames
)
from musicaiz.tokenizers import (
    MMMTokenizer,
    MMMTokenizerArguments,
    TokenizerArguments
)


class LakhMIDI(MusicGenerationDataset):
    """
    """

    def __init__(self):
        self.name = MusicGenerationDatasetNames.LAKH_MIDI.name.lower()

    def tokenize(
        self,
        dataset_path: Union[Path, str],
        output_path: Union[Path, str],
        tokenize_split: str,
        args: Type[TokenizerArguments],
        output_file: str = "token-sequences",
        train_split: float = 0.7,
        test_split: float = 0.2
    ) -> None:
        """

        Parameters
        ----------
        
        dataset_path (str): _description_

        output_path (str): _description_

        tokenize_split (str): _description_

        args (Type[TokenizerArguments]): _description_

        output_file (str, optional): _description_. Defaults to "token-sequences".
        
        Examples
        --------

        >>> # initialize tokenizer args
        >>> args = MMMTokenizerArguments(
        >>>    prev_tokens="",
        >>>    windowing=True,
        >>>    time_unit="HUNDRED_TWENTY_EIGHT",
        >>>    num_programs=None,
        >>>    shuffle_tracks=True,
        >>>    track_density=False,
        >>>    window_size=32,
        >>>    hop_length=16,
        >>>    time_sig=True,
        >>>    velocity=True,
        >>> )
        >>> # initialize dataset
        >>> dataset = LakhMIDI()
        >>> dataset.tokenize(
        >>>     dataset_path="path/to/dataset",
        >>>     output_path="output/path",
        >>>     output_file="token-sequences",
        >>>     args=args,
        >>>     tokenize_split="all"
        >>> )
        >>> # get vocabulary and save it in `dataset_path`
        >>> vocab = MMMTokenizer.get_vocabulary(
        >>>     dataset_path="output/path"
        >>> )
        """
        metadata = self.get_metadata(
            dataset_path,
            train_split,
            test_split
        )
        self._prepare_tokenize(
            dataset_path,
            output_path,
            output_file,
            metadata,
            tokenize_split,
            args,
            False
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
            composer = composer.replace(" ", "_")
            composer = composer.upper()

            # iterate over songs of an artist
            songs = [f for f in composer_path.glob("*/")]
            n_songs = len(songs)

            train_idxs = int(round(n_songs * train_split))
            val_idxs = int(n_songs * test_split)

            train_seqs = songs[:train_idxs]
            val_seqs = songs[train_idxs:val_idxs+train_idxs]


            # split in train, validation and test
            # we do this here to ensure that every artist is  at least in
            # the training and test sets (if n_songs > 1)
            for song in songs:
                #-----------------
                period = ""

                # 3. Process canonical genre
                genre = ""

                split = ""

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
                            "period": period,
                            "genre": genre,
                            "split": split
                        }
                    }
                )
        return composers_json


# TODO: args parsing here
if __name__ == "__main__":
    args = MMMTokenizerArguments(
        prev_tokens="",
        windowing=True,
        time_unit="HUNDRED_TWENTY_EIGHT",
        num_programs=None,
        shuffle_tracks=True,
        track_density=False,
        window_size=32,
        hop_length=16,
        time_sig=True,
        velocity=True,
    )
    dataset = LakhMIDI()
    dataset.tokenize(
        dataset_path="H:/INVESTIGACION/Datasets/LMD/clean_midi",
        output_path="H:/GitHub/musanalysis-datasets/lmd/mmm/32_bars_166",
        output_file="token-sequences",
        args=args,
        tokenize_split="validation"
    )
    vocab = MMMTokenizer.get_vocabulary(
        dataset_path="H:/GitHub/musanalysis-datasets/lmd/mmm/32_bars_166"
    )
