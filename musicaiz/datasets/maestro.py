from pathlib import Path
from typing import Type, Dict, Union
import pandas as pd
from enum import Enum

from musicaiz.datasets.configs import (
    MusicGenerationDataset,
    MusicGenerationDatasetNames
)
from musicaiz.tokenizers import (
    MMMTokenizer,
    MMMTokenizerArguments,
    TokenizerArguments,
)


_CANONICAL = [
    "SONATA",
    "PRELUDE",
    "PIANO_TRIO",
    "WALTZ",
    "ETUDE",
    "SCHERZO",
    "DANSE",
    "RHAPSODY",
    "SUITE",
    "CHACONNE",
    "SONATINA",
    "OPERA",
    "VARIATIONS",
    "RIGOLETTO",
    "TANGO",
    "FUGE",
    "PRELUDE_AND_FUGE",
    "OVERTURE",
    "PAVAN",
    "POLKA",
    "FANTASY",
    "POLONAISE",
    "BALLADE",
    "NOCTURNE"
]

class ComposerPeriods(Enum):
    ALBAN_BERG = "ROMANTICISM"
    ALEXANDER_SCRIABIN = "ROMANTICISM"
    ANTON_ARENSKY = "ROMANTICISM"
    ANTONIO_SOLER = "CLASSICISM"
    CARL_MARIA_VON_WEBER = "ROMANTICISM"
    CHARLES_GOUNOD = "ROMANTICISM"
    CLAUDE_DEBUSSY = "IMPRESSIONISM"
    CESAR_FRANCK = "ROMANTICISM"
    DOMENICO_SCARLATTI = "CLASSICISM"
    EDVARD_GRIEG = "ROMANTICISM"
    FELIX_MENDELSSOHN = "ROMANTICISM"
    FRANZ_LISTZ = "ROMANTICISM"
    FRANZ_SCHUBERT = "CLASSICISM"
    FRITZ_KREISLER = "CLASSICISM"
    FREDERIC_CHOPIN = "ROMANTICISM"
    GEORGE_ENESCU = "ROMANTICISM"
    GEORGE_FRIDERIC_HANDEL = "BAROQUE"
    GEORGES_BIZET = "ROMANTICISM"
    GIUSEPPE_VERDI = "ROMANTICISM"
    HENRY_PURCELL = "BAROQUE"
    ISAAC_ALBENIZ = "ROMANTICISM"
    JEAN_PHILIPPE_RAMEAU = "BAROQUE"
    JOHANN_CHISTIAN_FISCHER = "CLASSICISM"
    JOHANN_PACHELBEL = "BAROQUE"
    JOHANN_SEBASTIAN_BACH = "BAROQUE"
    JOHANN_STRAUSS = "ROMANTICISM"
    JOHANNES_BRAHMS = "ROMANTICISM"
    JOSEP_HAYDN = "CLASSICISM"
    LEOS_JANACEK = "ROMANTICISM"
    LUDWIG_VAN_BEETHOVEN = "CLASSICISM"
    MIKHAIL_GLINKA = "ROMANTICISM"
    MILY_BALARIKEV = "ROMANTICISM"  # TODO: MILY_BALAKIREV
    MODEST_MUSSORGSKY = "ROMANTICISM"
    MUZIO_CLEMENTI = "CLASSICISM"
    NICCOLO_PAGANINI = "CLASSICISM"
    NIKOLAI_MEDTNER = "ROMANTICISM"
    NIKOLAI_RIMSKY_KORSAKOV = "ROMANTICISM"
    ORLANDO_GIBBONS = "BAROQUE"
    PERCY_GRAIGNER = "ROMANTICISM"
    PYOTR_ILYICH_TCHAIKOVSKY = "ROMANTICISM"
    RICHARD_WAGNER = "ROMANTICISM"
    ROBERT_SCHUMANN = "ROMANTICISM"
    SERGEI_RACHMANINOFF = "ROMANTICISM"
    WOLFGANG_AMADEUS_MOZART = "CLASSICISM"


class Maestro(MusicGenerationDataset):
    """
    This class contains methods to process the Maestro dataset:
    *Hawthorne, C., Stasyuk, A., Roberts, A., Simon, I., Huang, C. Z. A., Dieleman, S., ... & Eck, D. (2018).
    Enabling factorized piano music modeling and generation with the MAESTRO dataset.
    arXiv preprint arXiv:1810.12247.*
    """
    def __init__(self):
        self.name = MusicGenerationDatasetNames.MAESTRO.name.lower()
    
    def tokenize(
        self,
        dataset_path: Union[str, Path],
        output_path: str,
        tokenize_split: str,
        args: Type[TokenizerArguments],
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
        >>> dataset = Maestro()
        >>> dataset.tokenize(
        >>>     dataset_path="path/to/dataset",
        >>>     output_path="output/path",
        >>>     output_file="token-sequences",
        >>>     args=args,
        >>>     tokenize_split="train"
        >>> )
        >>> # get vocabulary and save it in `dataset_path`
        >>> vocab = MMMTokenizer.get_vocabulary(
        >>>     dataset_path="output/path"
        >>> )
        """
        dataset_path = str(Path(dataset_path, "maestro-v2.0.0"))
        metadata = self.get_metadata(dataset_path)
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
    def get_metadata(dataset_path: Union[str, Path]) -> Dict[str, str]:
        """Prepares the metadata json from the Maestro csv."""
        table = pd.read_csv(str(Path(dataset_path, "maestro-v2.0.0.csv")))
        composers_json = {}
        for index, row in table.iterrows():
        # 1. Process composer
            composer = row["canonical_composer"]
            # Some composers are written with 2 different composers separated by "/"
            # we'll only consider the 1st one
            composer = composer.split("/")[0]
            composer = composer.replace(" ", "_")
            composer = composer.upper()
            if composer not in ComposerPeriods.__members__.keys():
                continue

            # 2. Process period
            period = ComposerPeriods[composer].value

            # 3. Process canonical genre
            genre = row["canonical_title"]
            for item in _CANONICAL:
                if item not in genre.upper():
                    continue
                canonical = item
            composers_json.update(
                {
                    row["midi_filename"]: {
                        "composer": composer,
                        "period": period,
                        "genre": canonical,
                        "split": row["split"]
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
        tempo=True,
    )
    dataset = Maestro()
    dataset.tokenize(
        dataset_path="H:/INVESTIGACION/Datasets/MAESTRO/",
        output_path="H:/GitHub/musanalysis-datasets/maestro/mmm/32_bars_16",
        output_file="token-sequences",
        args=args,
        tokenize_split="all"
    )
    vocab = MMMTokenizer.get_vocabulary(
        dataset_path="H:/GitHub/musanalysis-datasets/maestro/mmm/32_bars_16"
    )
