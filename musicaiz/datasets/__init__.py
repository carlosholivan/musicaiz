"""
Datasets
========

This submodule presents helper functions to works and process MIR datasets for
different tasks such as:

    - Automatic Chord Recognition: ACR
    - Music Generation or Composition

.. note::
    Not all the datasets shown here are included in `musicaiz` souce code.
    Some of these datasets have their own GitHub repository with their corresponding helper functions.



Music Composition or Generation
-------------------------------

    - JS Bach Chorales

        Boulanger-Lewandowski, N., Bengio, Y., & Vincent, P. (2012).
        Modeling temporal dependencies in high-dimensional sequences:
        Application to polyphonic music generation and transcription.
        arXiv preprint arXiv:1206.6392.

        | Download: http://www-ens.iro.umontreal.ca/~boulanni/icml2012
        | Paper: https://arxiv.org/abs/1206.6392


    - JS Bach Fakes

        Peracha, O. (2021).
        Js fake chorales: a synthetic dataset of polyphonic music with human annotation.
        arXiv preprint arXiv:2107.10388.

        | Paper: https://arxiv.org/abs/2107.10388
        | Repository & Download: https://github.com/omarperacha/js-fakes


    - Lakh MIDI Dataset (LMD)

        Raffel, C. (2016).
        Learning-based methods for comparing sequences, with applications to audio-to-midi alignment and matching.
        Columbia University.

        | Thesis: http://colinraffel.com/publications/thesis.pdf
        | Download: https://colinraffel.com/projects/lmd/


    - MAESTRO

        Hawthorne, C., Stasyuk, A., Roberts, A., Simon, I., Huang, C. Z. A., Dieleman, S., ... & Eck, D. (2018).
        Enabling factorized piano music modeling and generation with the MAESTRO dataset. arXiv preprint arXiv:1810.12247.

        | Download: https://magenta.tensorflow.org/datasets/maestro
        | Paper: https://arxiv.org/abs/1810.12247


    - Slakh2100

        Manilow, E., Wichern, G., Seetharaman, P., & Le Roux, J. (2019).
        Cutting music source separation some Slakh: A dataset to study the impact of training data quality and quantity.
        In 2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA) (pp. 45-49). IEEE.

        | Download link: https://zenodo.org/record/4599666#.YpD8ZO7P1PY
        | Paper: https://ieeexplore.ieee.org/abstract/document/8937170
        | Repository: https://github.com/ethman/Slakh


    - Meta-MIDI Dataset

        Ens, J., & Pasquier, P. (2021).
        Building the metamidi dataset: Linking symbolic and audio musical data.
        In Proceedings of 22st International Conference on Music Information Retrieval, ISMIR.

        | Download link: https://zenodo.org/record/5142664#.YpD8he7P1PY
        | Paper: https://archives.ismir.net/ismir2021/paper/000022.pdf
        | Repository: https://github.com/jeffreyjohnens/MetaMIDIDataset


Automatic Chord Recognition
---------------------------

    - Schubert Winterreise Dataset
    
        Christof Weiß, Frank Zalkow, Vlora Arifi-Müller, Meinard Müller, Hendrik Vincent Koops, Anja Volk, & Harald G. Grohganz.
        (2020). Schubert Winterreise Dataset [Data set]. In ACM Journal on Computing and Cultural Heritage (1.0).

        | Download: https://doi.org/10.5281/zenodo.3968389
        | Paper: https://dl.acm.org/doi/10.1145/3429743

        .. autosummary::
            :toctree: generated/

            SWDPathsConfig
            SWD_FILES
            shubert_winterreise
    

    - BPS-FH Dataset
    
        Chen, T. P., & Su, L. (2018).
        Functional Harmony Recognition of Symbolic Music Data with Multi-task Recurrent Neural Networks.
        In Proceedings of 19th International Conference on Music Information Retrieval, ISMIR. pp. 90-97.

        | Repository & Download: https://github.com/Tsung-Ping/functional-harmony
        | Paper: http://ismir2018.ircam.fr/doc/pdfs/178_Paper.pdf

        .. autosummary::
            :toctree: generated/

            BPSFHPathsConfig

"""

from .configs import (
    MusicGenerationDataset,
)
from .jsbchorales import JSBChorales
from .lmd import LakhMIDI
from .maestro import Maestro
from .bps_fh import BPSFH


__all__ = [
    "MusicGenerationDataset",
    "JSBChorales",
    "LakhMIDI",
    "Maestro",
    "BPSFH"
]
