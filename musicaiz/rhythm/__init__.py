"""
Rhythm
======

This module provides objects and methods that define and deal with time events.

The submodule is divided in:

- Key: A tonic and mode (additionally it can have a chord progresison as attribute)

- Chord Progression: A list of chords

- Chord: List of 2 intervals (triad chords), 3 intervals (7ths), etc.

- Interval: List of 2 notes.

Timing
------

Defines and contains helper functions to deal with Time Signatures, time events, etc.

.. autosummary::
    :toctree: generated/

    TimingConsts
    NoteLengths
    ms_per_tick
    _bar_str_to_tuple
    ticks_per_bar
    ms_per_note
    get_subdivisions
    TimeSignature


Quantizer
---------

Quantizes symbolic music as it is done in Logic Pro by following the steps
described in:

[1] https://www.fransabsil.nl/archpdf/advquant.pdf

.. autosummary::
    :toctree: generated/

    QuantizerConfig
    basic_quantizer
    advanced_quantizer
    get_ticks_from_subdivision
    _find_nearest

"""

from .timing import (
    TimingConsts,
    NoteLengths,
    SymbolicNoteLengths,
    TimeSignature,
    ms_per_tick,
    _bar_str_to_tuple,
    ticks_per_bar,
    ms_per_note,
    ms_per_bar,
    get_subdivisions,
    get_symbolic_duration,
    Timing,
    Beat,
    Subdivision,
)

from .quantizer import (
    QuantizerConfig,
    basic_quantizer,
    advanced_quantizer,
    get_ticks_from_subdivision,
    _find_nearest
)

__all__ = [
    "TimingConsts",
    "NoteLengths",
    "ms_per_tick",
    "_bar_str_to_tuple",
    "ticks_per_bar",
    "ms_per_note",
    "ms_per_bar",
    "get_subdivisions",
    "QuantizerConfig",
    "basic_quantizer",
    "advanced_quantizer",
    "get_ticks_from_subdivision",
    "_find_nearest",
    "SymbolicNoteLengths",
    "get_symbolic_duration",
    "TimeSignature",
    "Timing",
    "Beat",
    "Subdivision",
]
