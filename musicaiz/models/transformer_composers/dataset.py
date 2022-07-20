import os
from pathlib import Path
from typing import List, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader


def build_torch_loaders(
    dataset_path: Union[str, Path],
    sequence_length: int,
    batch_size: int,
    train_split: float = 0.9,
    is_splitted: bool = False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Builds the train a validation dataloaders.

    Parameters
    ----------

    dataset_path: Path

    sequence_length: int

    batch_size: int

    dest_path: Union[str, Path]
        The destination path if `save=True`.

    train_split: float. Between 0 and 1.
        The training...
    
    save: bool
        If we want to save in disk the splitted tokens seqs.
    
    is_splitted: bool.
        Default is False.
        If the dataset is already splitted in train and validation (and test) sets,
        and there's one `token-sequences.txt` file in each directory, it reads
        the token sequences and builds the loaders with them and it won't split the
        files automatically.
    
    Returns
    -------

    train_dataloader: torch.utils.data.Dataloader
        The train loader.
    
    val_dataloader: torch.utils.data.Dataloader
        The validation loader.
    """
    # TODO: Implement save token seqs if save is True
    if train_split >= 1:
        raise Exception(f"Training set must be between 0 and 1.")

    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)
    
    if is_splitted:
        vocab_path = Path(dataset_path)
        train_path = Path(dataset_path, "train")
        val_path = Path(dataset_path, "validation")

        train_token_seqs = tokens_to_sequences(train_path, vocab_path, sequence_length)
        val_token_seqs = tokens_to_sequences(val_path, vocab_path, sequence_length)

        train_data = MIDIDataset(train_token_seqs)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        val_data = MIDIDataset(val_token_seqs)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    else:
        train_path = dataset_path
        val_path = dataset_path
        vocab_path = dataset_path
    
        train_token_seqs = tokens_to_sequences(train_path, vocab_path, sequence_length)

        train_idxs = int(len(train_token_seqs) * train_split)

        train_seqs = list(train_token_seqs)[:train_idxs]
        val_seqs = list(train_token_seqs)[train_idxs:]

        train_data = MIDIDataset(train_seqs)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        val_data = MIDIDataset(val_seqs)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader


class MIDIDataset(Dataset):

    def __init__(
        self,
        token_seqs: List[List[int]],
    ):
        self.token_seqs = token_seqs

    def __len__(self):
        return len(self.token_seqs)

    def __getitem__(self, idx):
        sequence = torch.IntTensor(self.token_seqs[idx])
        return sequence


def tokens_to_sequences(
    dataset_path: Union[str, Path],
    vocab_path: Union[str, Path],
    sequence_length: int
):
    """
    Converts the token sequences stored in .txt files to a seuqnce of ints with
    the vocabulary stored in vocabulary.txt.

    Parameters
    ----------

    dataset_path: Path
        The path with  the tokens txt files and the vocabulary.txt file.
    
    sequence_length: int
        The length of the sequence.

    Returns
    -------

    indices_output_list: List[List[int]]
        A list in which each item is the list of ints corresponding to the token
        indices of a piece.

    Raises
    ------

    Exception: _description_

    Exception: _description_
    """
    # Raise an exception if path does not exist.
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)
    if not os.path.exists(dataset_path):
        raise Exception(f"Path does not exist. {dataset_path}")

    # Glob all the textfiles that are in the path and that start with "token-sequences".
    text_files = list(dataset_path.glob("*.txt"))

    # Throw an expection if there are no text files.
    if len(text_files) == 0:
        raise Exception(f"No text files in the path. {dataset_path}")

    # Get the vobabulary: This will serve to map the current token sequences to
    # ints that corresponds to the vocabulary items
    tokens = get_vocabulary(vocab_path)

    # Go through all the textfiles and create a dataset.
    indices_input_list = []
    for text_file in text_files:

        # don't consider vocabulary.txt file as a token sequence for training
        if "vocabulary" in text_file.stem:
            continue
        
        # Log the file. Only pring basename
        print(f"Loading text file: {os.path.basename(text_file)}")

        # Go through the text file line by line.
        for line in open(text_file, "r"):
                
            # Split the line by whitespace.
            line_tokens = line.split()
            if len(line_tokens) == 0:
                continue

            # Create a list of integers.
            tokens = get_vocabulary(vocab_path)
            line_tokens_indices = [tokens.index(token) for token in line_tokens]

            # Split a piece in the sequence_length
            # If the length of the line is less than the sequence length plus one, pad the line.
            if len(line_tokens_indices) < sequence_length + 1:
                line_tokens_indices = line_tokens_indices + [tokens.index("PAD")] * (sequence_length + 1 - len(line_tokens_indices))

            # If the length of the line is more than the sequence length plus one, truncate the line.
            if len(line_tokens_indices) > sequence_length + 1:
                line_tokens_indices = line_tokens_indices[:sequence_length + 1]

            # Check if everything is fine.
            assert len(line_tokens_indices) == sequence_length + 1

            # Get the input and output indices.
            indices_input = line_tokens_indices[:-1]
            assert len(indices_input) == sequence_length

            # Append to list.
            indices_input_list.append(indices_input)

        # Log the number of samples.
        print(f"Number of pieces {len(indices_input_list)} of length {sequence_length} in path {dataset_path}")
    return indices_input_list


def get_vocabulary(dataset_path: Union[str, Path]) -> List[str]:
    """
    Read one txt file and retrieves the vocabulary.
    In the ``dataset_path`` directory there must be a ``XX_vocabulary.txt`` file where
    the vocabulary is stored.

    Parameters
    ----------

    dataset_path: Path
        The path where the .txt files with token sequences and vocaulary are.
    
    Returns
    -------

    tokens: List[str]
    """
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)

    tokens_file = list(dataset_path.glob("*vocabulary.txt"))
    if len(tokens_file) == 0:
        raise Exception(f"No ``vocabulary.txt`` found in {dataset_path}.")
    tokens_file_string = open(tokens_file[0], 'r').read()
    tokens = ["PAD"] + tokens_file_string.split()
    return tokens
