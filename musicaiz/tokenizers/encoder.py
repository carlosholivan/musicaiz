from __future__ import annotations
from abc import ABCMeta
from typing import List, Dict, Type, Union
from pathlib import Path
from datetime import datetime
import dataclasses
import json


class TokenizerArguments(metaclass=ABCMeta):

    @staticmethod
    def save(
        args: Type[TokenizerArguments],
        out_dir: Union[str, Path],
        file: str = "configs.json",
    ):
        """
        Saves the configs as a json file.
        """
        d = dataclasses.asdict(args)
        # add datetime to json
        d["datetime"] = str(datetime.now())
        with open(Path(out_dir, file), 'w') as fp:
            json.dump(d, fp)


class EncodeBase(metaclass=ABCMeta):

    @classmethod
    def _get_tokens_analytics(
        cls,
        tokens: str,
        note_token: str,
        piece_start_token: str,
    ) -> Dict[str, int]:
        """
        Extracts features to aanlyze the given token sequence.

        Parameters
        ----------

        tokens: str
            A token sequence.

        Returns
        -------

        analytics: Dict[str, int]
            The ``analytics`` dict keys are:
                - ``total_tokens``
                - ``unique_tokens``
                - ``total_notes``
                - ``unique_notes``
                - ``total_bars``: non empty bars
                - ``total_instruments``
                - ``unique_instruments``
        """
        # Convert str in list of pieces that contain tokens
        # We suppose that the piece starts with a BAR=0 token (that is, any instr has notes in the 1st bar)
        dataset_tokens = cls._get_pieces_tokens(tokens, piece_start_token)
        # Start the analysis
        note_counts, bar_counts, instr_counts = 0, 0, 0  # total notes and bars (also repeated note values)
        total_toks = 0
        unique_tokens, unique_notes, unique_instr = [], [], []  # total non-repeated tokens
        unique_genres, unique_composers, unique_periods = [], [], []
        for piece, toks in enumerate(dataset_tokens):
            for tok in toks:
                total_toks += 1
                if tok not in unique_tokens:
                    unique_tokens.append(tok)
                if note_token in tok:
                    note_counts += 1
                if "BAR" in tok:
                    bar_counts += 1
                if "INST" in tok:
                    instr_counts += 1
                if note_token in tok and tok not in unique_notes:
                    unique_notes.append(tok)
                if "INST" in tok and tok not in unique_instr:
                    unique_instr.append(tok)
                if "GENRE" in tok and tok not in unique_genres:
                    unique_genres.append(tok)
                if "PERIOD" in tok and tok not in unique_periods:
                    unique_periods.append(tok)
                if "COMPOSER" in tok and tok not in unique_composers:
                    unique_composers.append(tok)
        if piece_start_token == "BAR=0":
            bar_counts += 1
        analytics = {
            "total_pieces": piece + 1,
            "total_tokens": len(tokens.split(" ")),
            "unique_tokens": len(unique_tokens),
            "total_notes": note_counts,
            "unique_notes": len(unique_notes),
            "total_bars": bar_counts,
            "total_instruments": instr_counts,
        }
        if len(unique_genres) != 0:
            analytics.update({"unique_genres": len(unique_genres)})
        if len(unique_periods) != 0:
            analytics.update({"unique_periods": len(unique_periods)})
        if len(unique_composers) != 0:
            analytics.update({"unique_composers": len(unique_composers)})

        return analytics

    @staticmethod
    def _get_pieces_tokens(tokens: str, token: str) -> List[List[str]]:
        """Converts the tokens str that can contain one or more
        pieces into a list of pieces that are also lists which contain
        one item per token.

        Example (MMMTokenizer)
        ----------------------
        >>> tokens = "PIECE_START INST=0 ... PIECE_START ..."
        >>> dataset_tokens = _get_pieces_tokens(tokens, "PIECE_START")
        >>> [
                ["PIECE_START INST=0 ...],
                ["PIECE_START ...],
            ]
        """
        tokens = tokens.split(token)
        if "" in tokens: tokens.remove("")
        dataset_tokens = []
        for piece in tokens:
            piece_tokens = piece.split(" ")
            if "" in piece_tokens: piece_tokens.remove("")
            dataset_tokens.append(piece_tokens)
        return dataset_tokens

    @staticmethod
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
