![plot](docs/images/logo_rectangle.png?raw=true)

# MUSICAIZ

A Python library for symbolic music generation, analysis and visualization.

<!-- SHIELDS -->
<!-- markdownlint-disable -->
<table>
  <colgroup>
    <col style="width: 10%;"/>
    <col style="width: 90%;"/>
  </colgroup>
  <tbody>
    <tr>
      <th>CI</th>
      <td>
        <img alt="build" src="https://github.com/carlosholivan/musicaiz/actions/workflows/ci.yml/badge.svg"/>
        </a>
      </td>
    </tr>
    <tr>
      <th>PyPI</th>
      <td>
        <a href="https://pypi.org/project/PyFLP">
          <img alt="PyPI - Package Version" src="https://img.shields.io/pypi/v/musicaiz"/>
        </a>
        <a href="https://pypi.org/project/musicaiz">
          <img alt="PyPI - Supported Python Versions" src="https://img.shields.io/pypi/pyversions/musicaiz?logo=python&amp;logoColor=white"/>
        </a>
        <a href="https://pypi.org/project/musicaiz">
          <img alt="PyPI - Supported Implementations" src="https://img.shields.io/pypi/implementation/musicaiz"/>
        </a>
        <a href="https://pypi.org/project/PyFLP">
          <img alt="PyPI - Wheel" src="https://img.shields.io/pypi/wheel/musicaiz"/>
        </a>
      </td>
    </tr>
    <tr>
      <th>Activity</th>
      <td>
        <img alt="Maintenance" src="https://img.shields.io/maintenance/yes/2022"/>
      </td>
    </tr>
    <tr>
      <th>QA</th>
      <td>
        <a href="https://codecov.io/gh/carlosholivan/musicaiz">
          <img alt="codecov" src="https://codecov.io/gh/carlosholivan/musicaiz/branch/main/graph/badge.svg?token=RGSRMMF8PF"/>
        </a>
        <a href="https://github.com/pre-commit/pre-commit">
          <img alt="pre-commit" src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&amp;logoColor=white"/>
        </a>
      </td>
    </tr>
    <tr>
      <th>Code</th>
      <td>
        <a href="https://github.com/demberto/PyFLP/blob/master/LICENSE">
          <img alt="License" src="https://img.shields.io/github/license/carlosholivan/musicaiz"/>
        </a>
        <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/carlosholivan/musicaiz"/>
        <a href="https://github.com/psf/black">
          <img alt="Code Style: Black" src="https://img.shields.io/badge/code%20style-black-black"/>
        </a>
      </td>
    </tr>
  </tbody>
</table>


[See the docs](https://carlosholivan.github.io/musicaiz)

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
@misc{hernandezolivan22musicaiz,
  doi = {10.48550/ARXIV.2209.07974},
  url = {https://arxiv.org/abs/2209.07974},
  author = {Hernandez-Olivan, Carlos and Beltran, Jose R.},
  title = {musicaiz: A Python Library for Symbolic Music Generation, Analysis and Visualization},
  publisher = {arXiv},
  year = {2022},
}
````

## Contributing

Musicaiz software can be extended in different ways, see some example in [TODOs](TODOs.md). If you want to contribute, please follow the guidelines in [Develop](##Develop)
