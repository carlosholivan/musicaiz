"""
Tokenizers
==========

This module provides methods to encode symbolic music in order to train Sequence models.

The parent class in the EncodeBase class. The tokenizers are:

    - MMMTokenizer: Multi-Track Music Machine tokenizer.


Basic Encoding
--------------

.. autosummary::
    :toctree: generated/

    EncodeBase
    TokenizerArguments


Multi-Track Music Machine
-------------------------

This submodule contains the implementation of the MMM encoding:

[1] Ens, J., & Pasquier, P. (2020).
Flexible generation with the multi-track music machine.
In Proceedings of the 21st International Society for Music Information Retrieval Conference, ISMIR.

.. autosummary::
    :toctree: generated/

    MMMTokenizerArguments
    MMMTokenizer


Multi-Track Music Machine
-------------------------

This submodule contains the implementation of the REMI encoding:

[2] Huang, Y. S., & Yang, Y. H. (2020).
Pop music transformer: Beat-based modeling and generation of expressive pop piano compositions.
In Proceedings of the 28th ACM International Conference on Multimedia (pp. 1180-1188).

.. autosummary::
    :toctree: generated/

    REMITokenizerArguments
    REMITokenizer

"""

from enum import Enum

from .encoder import (
    EncodeBase,
    TokenizerArguments,
)
from .mmm import (
    MMMTokenizer,
    MMMTokenizerArguments,
)
from .remi import (
    REMITokenizer,
    REMITokenizerArguments,
)
from .one_hot import (
    OneHot,
)


TOKENIZER_ARGUMENTS = [
    MMMTokenizerArguments,
    REMITokenizerArguments,
]


class Tokenizers(Enum):
    MULTI_TRACK_MUSIC_MACHINE = ("MMM", MMMTokenizerArguments)
    REMI = ("REMI", REMITokenizerArguments)

    @property
    def name(self):
        return self.value[0]
    
    @property
    def arg(self):
        return self.value[1]

    @staticmethod
    def names():
        return [t.value[0] for t in Tokenizers.__members__.values()]

    @staticmethod    
    def args():
        return [t.value[1] for t in Tokenizers.__members__.values()]


__all__ = [
    "EncodeBase",
    "TokenizerArguments",
    "TOKENIZER_ARGUMENTS",
    "Tokenizers",
    "MMMTokenizerArguments",
    "MMMTokenizer",
    "REMITokenizerArguments",
    "REMITokenizer",
    "OneHot"
]
