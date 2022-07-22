![plot](docs/images/logo_rectangle.png?raw=true)

# MUSICAIZ

A Python library for symbolic music generation, analysis and visualization.

[![PyPI](https://img.shields.io/pypi/v/musicaiz.svg)](https://pypi.python.org/pypi/musicaiz)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/musicaiz)](https://pypi.org/project/musicaiz) [![Supported Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Windows%20%7C%20Linux-green)](https://pypi.org/project/musanalysis) [![PyPI - Downloads](https://img.shields.io/pypi/dm/musicaiz)](https://pypistats.org/packages/musicaiz)

[![CI](https://github.com/carlosholivan/musicaiz/actions/workflows/ci.yml/badge.svg)](https://github.com/carlosholivan/musicaiz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/carlosholivan/musicaiz/branch/main/graph/badge.svg?token=ULWnUHaIJC)](https://codecov.io/gh/carlosholivan/musicaiz)


See the docs [here](https://carlosholivan.github.io/musicaiz)

The modules contained in this library are:

- [Structure](musicaiz/structure/)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains the structure elements in music (instruments, bars and notes).
- [Harmony](musicaiz/harmony/)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains the harmonic elements in music (intervals, chords and keys).
- [Rhythm](musicaiz/rhythm/)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains rhythmic or timing elements in music (quantization).
- [Features](musicaiz/features/)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains classic features to analyze symbolic music data (pitch class histograms...).
- [Algorithms](musicaiz/algorithms/)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains algorithms for chord prediction, key prediction, harmonic transposition...
- [Plotters](musicaiz/plotters/)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains different ways of plotting music (pinorolls or scores).
- [Tokenizers](musicaiz/tokenizers/)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains different encodings to prepare symbolic music data to train a sequence model.
- [Converters](musicaiz/harmony/)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains converters to other formats (JSON,...).
- [Datasets](musicaiz/datasets/)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains helper methods to work with MIR open-source datasets.
- [Models](musicaiz/models/)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains ML models to generate symbolic music.

## License

This project is licensed under the terms of the [AGPL v3 license](LICENSE).


## Install

To install the latest stable version run: `pip install musicaiz`

To install the latest version, clone this repository and run:

`pip install -e .`

If you want to train the models in the [models](musicaiz/models/) submodule, you must install `apex`. Follow the instructions on https://github.com/NVIDIA/apex.


## Develop

### Conda dev environment

`conda env update -f environment.yml`

`conda activate musicaiz`

### Linting

flake8 and black

### Typing

Use mypy package to check variables tpyes:

`mypy musicaiz`

## Examples

See docs.

## Citing

If you use this software for your research, please cite:

````
@article{HernandezOlivan22,
    author    = {
      Carlos Hernandez-Olivan and Jose Ramon Beltran},
    title = {musicaiz: A Python Library for Symbolic Music
Generation, Analysis and Visualization},
    journal   = {XX},
    volume    = {x},
    number    = {x},
    pages     = {xx--xx},
    year      = {2022},
    url       = {XX},
    doi       = {XX},
}
````

## Contributing

Musicaiz software can be extended in different ways, see some example in [TODOs](TODOs.md). If you want to contribute, please follow the guidelines in [Develop](##Develop)
