from abc import ABCMeta, abstractmethod
from typing import List
from enum import Enum
from pathlib import Path


_POPULATE_NOTE_ON = [f"NOTE_ON={i} " for i in range(0, 128)]
_POPULATE_NOTE_OFF = [f"NOTE_OFF={i} " for i in range(0, 128)]
_POPULATE_NOTE_INST = [f"NOTE_INST={i} " for i in range(0, 128)]


class Tokens(Enum):
    GENRE = []
    SUBGENRE = []
    STRUCTURE = ["piece", "instrument", "bar"]
    NOTE_DENSITY = ["piece", "instrument", "bar"]
    HARMONIC_DENSITY = ["piece", "instrument", "bar"]
    CHORDS = ["piece", "instrument", "bar", "time_step"]


class TokenizerArguments:
    pass


class EncodeBase(metaclass=ABCMeta):

    def get_vocabulary(
        dataset_path: str,
        vocab_filename: str = "vocabulary"
    ) -> List[str]:
        """This method gets the vocabulary of a tokenize dataset in all the
        `token-sequences.txt` files in the directory `dataset_path`.
        The method will generate a `vocabulary.txt` file in the dataset path that
        will contain all the unique tokens in the dataset txt sequences.
        """
        # if dataset is splitte in train, validation and test dirs, we
        files = [file for file in Path(dataset_path).rglob("*token-sequences.txt")]

        if len(files) == 0:
            raise ValueError(f"Tokens sequences files not found in {dataset_path}. Could not process tokens.")

        vocabulary = []
        for file in files:
            print(f"Processing file...{file}")
            with open(file, "r") as txt_file:
                lines = txt_file.readlines()
                lines = [line.rstrip() for line in lines]
                for l, line in enumerate(lines):
                    line = line.split(" ")
                    for token in line:
                        if token not in vocabulary:
                            vocabulary.append(token)
                        if token == " ":
                            continue

        with open(Path(dataset_path, vocab_filename + ".txt"), "w") as vocab_file:
            vocab_file.write(" ".join(vocabulary))
        
        return vocabulary
    
    @staticmethod
    def add_token_to_vocabulary():
        pass

    @staticmethod
    def _to_file(
        all_files_tokens: List[str],
        file_name: str,
        path: str,
        extension: str,
    ):
        full_path = path + file_name + extension
        with open(Path(full_path), "w") as f:
            for piece_tokens in all_files_tokens:
                f.write("%s\n" % piece_tokens)

    @classmethod
    def to_txt(
        cls,
        all_files_tokens: List[str],
        file_name: str,
        path: str,
    ):
        cls._to_file(all_files_tokens, file_name, path, ".txt")
