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


Multi-Track Music Machine
-------------------------

This submodule contains the implementation of the MMM encoding:

.. panels::

    [1] Ens, J., & Pasquier, P. (2020).
    Flexible generation with the multi-track music machine.
    In Proceedings of the 21st International Society for Music Information Retrieval Conference, ISMIR.

    [2] 


.. autosummary::
    :toctree: generated/

    TokenizerArguments
    MMMTokenizerArguments
    MMMTokenizer
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
