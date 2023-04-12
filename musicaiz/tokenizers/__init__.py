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


REMI and REMI+
--------------

This submodule contains the implementation of the REMI encoding:

[2] Huang, Y. S., & Yang, Y. H. (2020).
Pop music transformer: Beat-based modeling and generation of expressive pop piano compositions.
In Proceedings of the 28th ACM International Conference on Multimedia (pp. 1180-1188).

And REMI+ encoding:

[3] von Rutte, D., Biggio, L., Kilcher, Y. & Hofmann, T. (2023).
FIGARO: Controllable Muic Generation using Learned and Expert Features.
ICLR 2023.

.. autosummary::
    :toctree: generated/

    REMITokenizerArguments
    REMITokenizer

Compound Word (CPWord)
--------------

This submodule contains the implementation of the CPWord encoding:

[4] Hsiao, W. Y., Liu, J. Y., Yeh, Y. C & Yang, Y. H. (2021).
Compund Word Transformer: Learning to compose full-song music over dynamic directed hypergraphs.
In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 35, No. 1, pp. 178-186).

.. autosummary::
    :toctree: generated/

    CPWordTokenizerArguments
    CPWordTokenizer
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
from .cpword import (
    CPWordTokenizerArguments,
    CPWordTokenizer,
)
from .one_hot import (
    OneHot,
)


TOKENIZER_ARGUMENTS = [
    MMMTokenizerArguments,
    REMITokenizerArguments,
    CPWordTokenizerArguments
]


class Tokenizers(Enum):
    MULTI_TRACK_MUSIC_MACHINE = ("MMM", MMMTokenizerArguments)
    REMI = ("REMI", REMITokenizerArguments)
    CPWORD = ("CPWORD", CPWordTokenizerArguments)

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
    "CPWordTokenizerArguments",
    "CPWordTokenizer",
    "OneHot"
]
