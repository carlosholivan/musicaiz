from pathlib import Path
from enum import Enum
from typing import List, Dict, Type
from abc import ABCMeta

from musicaiz.tokenizers import TokenizerArguments
from musicaiz.datasets.utils import tokenize_path


TOKENIZE_VALID_SPLITS = [
    "train",
    "validation",
    "test",
    "all",
]


class MusicGenerationDatasetNames(Enum):
    MAESTRO = ["maestro"]
    LAKH_MIDI = ["lakh_midi_dataset", "lakh_midi", "lmd"]
    JSB_CHORALES = ["jsbchorales", "jsb_chorales", "bach_chorales"]

    @classmethod
    def all_values(cls) -> List[str]:
        all = []
        for n in cls.__members__.values():
            for name in n.value:
                all.append(name)
        return all


class MusicGenerationDataset(metaclass=ABCMeta):

    def _prepare_tokenize(
        self,
        dataset_path: str,
        output_path: str,
        output_file: str,
        metadata: Dict[str, str],
        tokenize_split: str,
        args: Type[TokenizerArguments],
        dirs_splitted: bool,
    ) -> None:

        """Depending on the args that are passed to this method, the
        tokenization selected will be one of the available tokenizers that
        mathces with the args object.
        The current tokenizers available are: :const:`~musicaiz.tokenizers.constants.TOKENIZER_ARGUMENTS`
        """

        # make same dirs to store the token sequences separated in
        # train, valid and test
        dest_train_path = Path(output_path, "train")
        dest_train_path.mkdir(parents=True, exist_ok=True)

        dest_val_path = Path(output_path, "validation")
        dest_val_path.mkdir(parents=True, exist_ok=True)

        dest_test_path = Path(output_path, "test")
        dest_test_path.mkdir(parents=True, exist_ok=True)

        if metadata is not None:
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
        else:
            train_metadata, val_metadata, test_metadata = None, None, None

        if dirs_splitted:
            data_train_path = Path(dataset_path, "train")
            data_val_path = Path(dataset_path, "valid")
            data_test_path = Path(dataset_path, "test")
        else:
            data_train_path = dataset_path
            data_val_path = dataset_path
            data_test_path = dataset_path

        if tokenize_split not in TOKENIZE_VALID_SPLITS:
            raise ValueError(f"tokenize_split must be one of the following: {[f for f in TOKENIZE_VALID_SPLITS]}")
        if tokenize_split == "train" or tokenize_split == "all":
            tokenize_path(data_train_path, dest_train_path, train_metadata, output_file, args)
        if tokenize_split == "validation" or tokenize_split == "all":
            tokenize_path(data_val_path, dest_val_path, val_metadata, output_file, args)
        if tokenize_split == "test" or tokenize_split == "all":
            tokenize_path(data_test_path, dest_test_path, test_metadata, output_file, args)

        # save configs json
        TokenizerArguments.save(args, output_path)
