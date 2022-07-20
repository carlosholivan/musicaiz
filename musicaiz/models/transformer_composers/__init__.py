"""
Transformer Composers
=====================

This submodule presents a GPT2 model that generates music.

The tokenization is previously done with `musanalysis`
:func:`~musanalysis.tokenizers.MMMTokenizer` class.


Installation
------------

To train these models you should install torch with cuda. We recommend torch version 1.11.0
with cuda 113:

    >>> pip3 install torch==1.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

Apart from that, `apex` is also necessary. To install it properly, follow the instructions in: 
https://github.com/NVIDIA/apex


Configurations
--------------

.. autosummary::
    :toctree: generated/

    GPTConfigs
    TrainConfigs


Dataloaders
-----------

.. autosummary::
    :toctree: generated/

    build_torch_loaders
    get_vocabulary


Model
-----

.. autosummary::
    :toctree: generated/

    self_attention
    MultiheadAttention
    PositionalEncoding
    Embedding
    ResidualConnection
    FeedForward
    Decoder
    GPT2


Train
-----

.. autosummary::
    :toctree: generated/

    train

Generation
----------

.. autosummary::
    :toctree: generated/

    sample_sequence


Gradio App
----------

There's a simple app for this model built with Gradio.
To try the demo locally run:

>>> python models/transformer_composers/app.py


Examples
--------

Train model:

>>> python models/transformer_composers/train.py --dataset_path="..." --is_splitted True

Generate Sequence:

>>> python models/transformer_composers/generate.py --dataset_path H:/GitHub/musicaiz-datasets/jsbchorales/mmm/all_bars --dataset_name jsbchorales --save_midi True --file_path ../midi
"""

from .configs import (
    GPTConfigs,
    TrainConfigs
)
from .dataset import (
    build_torch_loaders,
    get_vocabulary
)
from .transformers import (
    self_attention,
    MultiheadAttention,
    PositionalEncoding,
    Embedding,
    ResidualConnection,
    FeedForward,
    Decoder,
    GPT2,
)
from .train import train
from .generate import sample_sequence


__all__ = [
    "GPTConfigs",
    "TrainConfigs",
    "build_torch_loaders",
    "get_vocabulary",
    "self_attention",
    "MultiheadAttention",
    "PositionalEncoding",
    "Embedding",
    "ResidualConnection",
    "FeedForward",
    "Decoder",
    "GPT2",
    "train",
    "sample_sequence",
]