"""
Models
======

This module provides baseline models for symbolic music generation.

The submodule is divided in:

- Transformer Composers: Transformer-based models.


Transformer Composers
---------------------

Contains a GPT model that can be trained to generate symbolic music.

.. autosummary::
    :toctree: generated/

    transformer_composers

"""


from . import (
    transformer_composers,
)

__all__ = [
    "transformer_composers",
]