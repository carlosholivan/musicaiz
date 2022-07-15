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


.. autosummary::
    :toctree: generated/

    MMMTokenizer

"""


from .encoder import (
    EncodeBase,
)
from .mmm import (
    MMMTokenizer,
)
from .one_hot import (
    OneHot,
)

__all__ = [
    "EncodeBase",
    "MMMTokenizer",
    "OneHot"
]
