"""
Structure
=========

This module provides objects and methods that allows to create and analyze the structure parts of
symbolic music.

The basic structure elements are:

- Piece: The whole piece or MIDI file that contains lists of instruments, bars, notes...
It can also contain harmonic attributes like Key, Chord Progressions, etc, depending if we do want
to predict or generate them.

- Instruments: A list of bars or directly, notes (depending if we want to distrubte the notes in
bars or not).

- Bar: List of notes (it can also contain a list of Chords).

- Notes: The basic element in music in both time and harmonic axes.

Notes
------

.. autosummary::
    :toctree: generated/

    AccidentalsNames
    AccidentalsValues
    NoteClassNames
    NoteClassBase
    NoteValue
    NoteTiming
    Note

Instruments
-----------

.. autosummary::
    :toctree: generated/

    InstrumentMidiPrograms
    InstrumentMidiFamilies
    Instrument

Bars
----

.. autosummary::
    :toctree: generated/

    Bar

"""

from .notes import (
    AccidentalsNames,
    AccidentalsValues,
    NoteClassNames,
    NoteClassBase,
    NoteValue,
    NoteTiming,
    Note,
)
from .instruments import (
    InstrumentMidiPrograms,
    InstrumentMidiFamilies,
    Instrument,
)
from .bars import Bar

__all__ = [
    "AccidentalsNames",
    "AccidentalsValues",
    "NoteClassNames",
    "NoteClassBase",
    "NoteValue",
    "NoteTiming",
    "Note",
    "InstrumentMidiPrograms",
    "InstrumentMidiFamilies",
    "Instrument",
    "Bar",
]
