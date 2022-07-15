"""
Harmony
========

This submodule contains objects that are related to harmony elements.

The basic harmonic elements are:

- Key: A tonic and mode (additionally it can have a chord progresison as attribute)

- Chord Progression: A list of chords

- Chord: List of 2 intervals (triad chords), 3 intervals (7ths), etc.

- Interval: List of 2 notes.

Intervals
---------

.. autosummary::
    :toctree: generated/

    IntervalClass
    IntervalQuality
    IntervalSemitones
    IntervalComplexity
    Interval

Chords
------

.. autosummary::
    :toctree: generated/

    ChordQualities
    ChordType
    AllChords
    Chord

Keys
----

.. autosummary::
    :toctree: generated/

    DegreesQualities
    DegreesRoman
    Degrees
    MajorTriadDegrees
    MinorNaturalTriadDegrees
    MinorHarmonicTriadDegrees
    MinorMelodicTriadDegrees
    DorianTriadDegrees
    PhrygianTriadDegrees
    LydianTriadDegrees
    MixolydianTriadDegrees
    LocrianTriadDegrees
    TriadsModes
    MajorSeventhDegrees
    MinorNaturalSeventhDegrees
    MinorHarmonicSeventhDegrees
    MinorMelodicSeventhDegrees
    DorianSeventhDegrees
    PhrygianSeventhDegrees
    LydianSeventhDegrees
    MixolydianSeventhDegrees
    LocrianSeventhDegrees
    SeventhsModes
    AccidentalNotes
    AccidentalDegrees
    ModeConstructors
    Scales
    Tonality
"""

from .intervals import (
    Interval,
    IntervalClass,
    IntervalQuality,
    IntervalSemitones,
    IntervalComplexity,
)
from .chords import (
    AllChords,
    ChordQualities,
    ChordType,
    Chord,
)
from .keys import (
    DegreesQualities,
    DegreesRoman,
    Degrees,
    MajorTriadDegrees,
    MinorNaturalTriadDegrees,
    MinorHarmonicTriadDegrees,
    MinorMelodicTriadDegrees,
    DorianTriadDegrees,
    PhrygianTriadDegrees,
    LydianTriadDegrees,
    MixolydianTriadDegrees,
    LocrianTriadDegrees,
    TriadsModes,
    MajorSeventhDegrees,
    MinorNaturalSeventhDegrees,
    MinorHarmonicSeventhDegrees,
    MinorMelodicSeventhDegrees,
    DorianSeventhDegrees,
    PhrygianSeventhDegrees,
    LydianSeventhDegrees,
    MixolydianSeventhDegrees,
    LocrianSeventhDegrees,
    SeventhsModes,
    AccidentalNotes,
    AccidentalDegrees,
    ModeConstructors,
    Scales,
    Tonality,
)

__all__ = [
    "IntervalClass",
    "IntervalQuality",
    "IntervalSemitones",
    "IntervalComplexity",
    "Interval",
    "ChordQualities",
    "ChordType",
    "AllChords",
    "Chord",
    "DegreesQualities",
    "DegreesRoman",
    "Degrees",
    "MajorTriadDegrees",
    "MinorNaturalTriadDegrees",
    "MinorHarmonicTriadDegrees",
    "MinorMelodicTriadDegrees",
    "DorianTriadDegrees",
    "PhrygianTriadDegrees",
    "LydianTriadDegrees",
    "MixolydianTriadDegrees",
    "LocrianTriadDegrees",
    "TriadsModes",
    "MajorSeventhDegrees",
    "MinorNaturalSeventhDegrees",
    "MinorHarmonicSeventhDegrees",
    "MinorMelodicSeventhDegrees",
    "DorianSeventhDegrees",
    "PhrygianSeventhDegrees",
    "LydianSeventhDegrees",
    "MixolydianSeventhDegrees",
    "LocrianSeventhDegrees",
    "SeventhsModes",
    "AccidentalNotes",
    "AccidentalDegrees",
    "ModeConstructors",
    "Scales",
    "Tonality",
]
