import warnings
import ruptures as rpt
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Union, Optional
from pathlib import Path
from enum import Enum

from musicaiz.loaders import Musa
from musicaiz.features import (
    get_novelty_func,
    musa_to_graph
)


LEVELS = ["high", "mid", "low"]
DATASETS = ["BPS", "SWD"]


@dataclass
class PeltArgs:
    penalty: int
    alpha: int
    betha: int
    level: str

class LevelsBPS(Enum):

    HIGH = PeltArgs(
        penalty = 4,
        alpha = 2.3,
        betha = 1.5,
        level = "high"
    )

    MID = PeltArgs(
        penalty = 0.5,
        alpha = 1,
        betha = 0.01,
        level = "mid"
    )

    LOW = PeltArgs(
        penalty = 0.1,
        alpha = 0.1,
        betha = 0.15,
        level = "low"
    )

class LevelsSWD(Enum):

    MID = PeltArgs(
        penalty = 0.7,
        alpha = 0.6,
        betha = 0.15,
        level = "mid"
    )

class StructurePrediction:

    def __init__(
        self,
        file: Optional[Union[str, Path]] = None,
    ):

        # Convert file into a Musa object to be processed
        if file is not None:
            self.midi_object = Musa(
                file=file,
            )
        else:
            self.midi_object = Musa(file=None)

    def notes(self, level: str, dataset: str) -> List[int]:
        return self._get_structure_boundaries(level, dataset)

    def beats(self, level: str, dataset: str) -> List[int]:
        result = self._get_structure_boundaries(level, dataset)
        return [self.midi_object.notes[n].beat_idx for n in result]

    def bars(self, level: str, dataset: str) -> List[int]:
        result = self._get_structure_boundaries(level, dataset)
        return [self.midi_object.notes[n].bar_idx for n in result]

    def ms(self, level: str, dataset: str) -> List[float]:
        result = self._get_structure_boundaries(level, dataset)
        return [self.midi_object.notes[n].start_sec * 1000 for n in result]

    def _get_structure_boundaries(
        self,
        level: str,
        dataset: str
    ):
        """
        Get the note indexes where a section ends.
        """
        if level not in LEVELS:
            raise ValueError(f"Level {level} not supported.")
        if dataset not in DATASETS:
            raise ValueError(f"Dataset {dataset} not supported.")
        if level == "high" and dataset == "BPS":
            pelt_args = LevelsBPS.HIGH.value
        elif level == "mid" and dataset == "BPS":
            pelt_args = LevelsBPS.MID.value
        elif level == "low" and dataset == "BPS":
            pelt_args = LevelsBPS.LOW.value
        elif level == "mid" and dataset == "SWD":
            pelt_args = LevelsSWD.MID.value
            
        g = musa_to_graph(self.midi_object)
        mat = nx.attr_matrix(g)[0]
        n = get_novelty_func(mat)
        nn = np.reshape(n, (n.size, 1))
        # detection
        try:
            algo = rpt.Pelt(
                model="rbf",
                min_size=pelt_args.alpha*(len(self.midi_object.notes)/15),
                jump=int(pelt_args.betha*pelt_args.alpha*(len(self.midi_object.notes)/15)),
            ).fit(nn)
            result = algo.predict(pen=pelt_args.penalty)
        except:
            warnings.warn("No structure found.")
            result = [0, len(self.midi_object.notes)-1]
        
        return result



