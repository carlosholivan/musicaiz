from pathlib import Path
from typing import Union, Callable
from abc import ABCMeta, abstractmethod

from musicaiz.tokenizers import MMMTokenizer


class MusicGenerationDataset(metaclass=ABCMeta):
    
    @abstractmethod
    def tokenize(
        dataset_path: Union[Path, str],
        output_path: Union[Path, str],
        tokenizer: str = "MMM",
        output_filename: str = "token-sequences",
    ):
        pass