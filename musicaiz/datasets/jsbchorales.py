from pathlib import Path
from typing import List, Union

from musicaiz.datasets.configs import (
    MusicGenerationDataset,
    MusicGenerationDatasetNames
)
from musicaiz.tokenizers import (
    MMMTokenizer,
    MMMTokenizerArguments,
)


class JSBChorales(MusicGenerationDataset):
    """
    """
    def __init__(self):
        self.name = MusicGenerationDatasetNames.JSB_CHORALES.name.lower()

    def tokenize(
        self,
        dataset_path: str,
        output_path: str,
        tokenize_split: str,
        args: MMMTokenizerArguments,
        output_file: str = "token-sequences",
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
        >>> dataset = JSBChorales()
        >>> dataset.tokenize(
        >>>     dataset_path="path/to/dataset",
        >>>     output_path="output/path",
        >>>     output_file="token-sequences",
        >>>     args=args,
        >>>     tokenize_split="test"
        >>> )
        >>> # get vocabulary and save it in `dataset_path`
        >>> vocab = MMMTokenizer.get_vocabulary(
        >>>     dataset_path="output/path"
        >>> )
        """

        self._prepare_tokenize(
            dataset_path,
            output_path,
            output_file,
            None,
            tokenize_split,
            args,
            True
        )


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
    dataset = JSBChorales()
    dataset.tokenize(
        dataset_path="H:/INVESTIGACION/Datasets/JSB Chorales/",
        output_path="H:/GitHub/musanalysis-datasets/jsbchorales/mmm/32_bars_166",
        output_file="token-sequences",
        args=args,
        tokenize_split="validation"
    )
    vocab = MMMTokenizer.get_vocabulary(
        dataset_path="H:/GitHub/musanalysis-datasets/jsbchorales/mmm/32_bars_166"
    )
