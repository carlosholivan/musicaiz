![plot](https://github.com/carlosholivan/musicaiz/blob/main/docs/images/logo_rectangle.png?raw=true)

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
        <a href="https://readthedocs.org/musicaiz/">
        <img alt="docs" src="https://readthedocs.org/projects/musicaiz/badge/?version=latest"/>
        </a>
      </td>
    </tr>
    <tr>
      <th>Paper</th>
      <td>
        <a href="https://arxiv.org/abs/2209.07974">
        <img alt= "arXiv" src="https://img.shields.io/badge/arXiv-1234.56789-00ff00.svg"/>
        </a>
      </td>
    </tr>
    <tr>
      <th>PyPI</th>
      <td>
        <a href="https://pypi.org/project/musicaiz/">
          <img alt="PyPI - Package Version" src="https://img.shields.io/pypi/v/musicaiz"/>
        </a>
        <a href="https://pypi.org/project/musicaiz">
          <img alt="PyPI - Supported Python Versions" src="https://img.shields.io/pypi/pyversions/musicaiz?logo=python&amp;logoColor=white"/>
        </a>
        <a href="https://pypi.org/project/musicaiz">
          <img alt="PyPI - Wheel" src="https://img.shields.io/pypi/wheel/musicaiz"/>
        </a>
        <a href="hhttps://pypistats.org/packages/musicaiz">
          <img alt="Downloads" src="https://img.shields.io/pypi/dm/musicaiz"/>
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
        <a href="https://api.codiga.io/project/34718/status/svg">
          <img alt="Code Quality" src="https://api.codiga.io/project/34718/status/svg"/>
        </a>
        <a href="https://api.codiga.io/project/34718/score/svg">
          <img alt="Code Score" src="https://api.codiga.io/project/34718/score/svg"/>
        </a>
        <a href="https://github.com/pre-commit/pre-commit">
          <img alt="pre-commit" src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&amp;logoColor=white"/>
        </a>
      </td>
    </tr>
    <tr>
      <th>Code</th>
      <td>
        <a href="https://github.com/carlosholivan/musicaiz/blob/master/LICENSE">
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

- [Loaders](musicaiz/loaders.py)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains the basic initialization to import files.

````python
from musicaiz.loaders import Musa

    midi = Musa(
      file="my_midifile.mid"
    )
````

- [Structure](musicaiz/structure/)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains the structure elements in music (instruments, bars and notes).

````python
# Define a Note object
from musicaiz.structure import Note

    note = Note(
      pitch=12,
      start=0.0,
      end=1.0,
      velocity=75,
      bpm=120,
      resolution=96
    )
````

- [Harmony](musicaiz/harmony/)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains the harmonic elements in music (intervals, chords and keys).

````python
from musicaiz.structure import Chords, Tonality

    # Initialize a chord by its notation
    chord_name = "Cm7b5"
    chord = Chord(chord_name)
    # get the notes in the chord
    chord.get_notes(inversion=0)

    # Initialize Tonality
    tonality = Tonality.E_MINOR
    # get different scales
    tonality.natural
    tonality.harmonic
    tonality.melodic
    # get the notes in a scale
    tonality.scale_notes("NATURAL")
    # get a chord from a scale degree
    Tonality.get_chord_from_degree(
      tonality="E_MINOR",
      degree="V",
      scale="NATURAL",
      chord_type="triad",
    )
````

- [Rhythm](musicaiz/rhythm/)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains rhythmic or timing elements in music (quantization).
- [Features](musicaiz/features/)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains classic features to analyze symbolic music data (pitch class histograms...).
- [Algorithms](musicaiz/algorithms/)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains algorithms for chord prediction, key prediction, harmonic transposition...
- [Plotters](musicaiz/plotters/)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains different ways of plotting music (pinorolls or scores).

````python
from musicaiz.plotters import Pianoroll, PianorollHTML

    # Matplotlib
    musa_obj = Musa(midi_sample)
    plot = Pianoroll(musa_obj)
    plot.plot_instruments(
        program=[48, 45],
        bar_start=0,
        bar_end=4,
        print_measure_data=True,
        show_bar_labels=False,
        show_grid=False,
        show=True,
    )

    # Pyplot HTML
    musa_obj = Musa(midi_sample)
    plot = PianorollHTML(musa_obj)
    plot.plot_instruments(
        program=[48, 45],
        bar_start=0,
        bar_end=4,
        show_grid=False,
        show=False
    )
````

- [Tokenizers](musicaiz/tokenizers/)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains different encodings to prepare symbolic music data to train a sequence model.

````python
from musicaiz.tokenizers import MMMTokenizer, MMMTokenizerArguments

    # Tokenize file
    midi = "my_midifile.mid"
    args = MMMTokenizerArguments(
      windowing=True,
      time_unit="SIXTEENTH",
      num_programs=None,
      shuffle_tracks=True,
      track_density=False,
      window_size=4,
      hop_length=1,
      time_sig=False,
      velocity=False,
      quantize=False,
      tempo=True,
    )
    # save configs
    MMMTokenizerArguments.save(args, "./")
    tokenizer = MMMTokenizer(midi, args)
    got = tokenizer.tokenize_file()

    # get tokens analysis
    my_tokens = "PIECE_START TRACK_START ..."
    MMMTokenizer.get_tokens_analytics(my_tokens)

    # Convert tokens to Musa objects
    MMMTokenizer.tokens_to_musa(
      tokens=my_tokens,
      absolute_timing=True,
      time_unit="SIXTEENTH",
      time_sig="4/4",
      resolution=96
    )

    # get vocabulary
    MMMTokenizer.get_vocabulary(
      dataset_path="apth/to/dataset/tokens",
    )
````

- [Converters](musicaiz/converters/)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains converters to other formats (JSON,...).

````python
from musicaiz.loaders import Musa
from musicaiz.loaders import musa_to_proto, proto_to_musa

  # Convert a musicaiz objects in protobufs
  midi = Musa(midi_sample, structure="bars")
  protobuf = musa_to_proto(midi)

  # Convert a protobuf to musicaiz objects
  musa = proto_to_musa(protobuf)
    
````

- [Datasets](musicaiz/datasets/)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;contains helper methods to work with MIR open-source datasets.

````python
from musicaiz.tokenizers import MMMTokenizer, MMMTokenizerArguments
from musicaiz.datasets import JSBChorales

    # Tokenize a dataset in musicaiz
    output_path = "path/to/store/tokens"

    args = MMMTokenizerArguments(
        prev_tokens="",
        windowing=True,
        time_unit="HUNDRED_TWENTY_EIGHT",
        num_programs=None,
        shuffle_tracks=True,
        track_density=False,
        window_size=32,
        hop_length=16,
        time_sig=True,
        velocity=True,
    )
    dataset = JSBChorales()
    dataset.tokenize(
        dataset_path="path/to/JSB Chorales/midifiles",
        output_path=output_path,
        output_file="token-sequences",
        args=args,
        tokenize_split="all"
    )
    vocab = MMMTokenizer.get_vocabulary(
        dataset_path=output_path
    )
````

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
