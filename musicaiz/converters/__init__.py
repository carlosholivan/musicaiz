"""
Converters
==========

This module allows to export symbolic music in different formats.


JSON
----

.. autosummary::
    :toctree: generated/

    MusaJSON


Pretty MIDI
-----------

.. autosummary::
    :toctree: generated/

    prettymidi_note_to_musicaiz
    musicaiz_note_to_prettymidi


Protobufs
----------

.. autosummary::
    :toctree: generated/

    musa_to_proto
    proto_to_musa
"""

from .musa_json import (
    MusaJSON,
    BarJSON,
    InstrumentJSON,
    NoteJSON,
)

from .pretty_midi_musa import (
    prettymidi_note_to_musicaiz,
    musicaiz_note_to_prettymidi,
    musa_to_prettymidi,
)

from .musa_protobuf import (
    musa_to_proto,
    proto_to_musa
)

from . import protobuf

__all__ = [
    "MusaJSON",
    "BarJSON",
    "InstrumentJSON",
    "NoteJSON",
    "prettymidi_note_to_musicaiz",
    "musicaiz_note_to_prettymidi",
    "musa_to_prettymidi",
    "protobuf",
    "musa_to_proto",
    "proto_to_musa"
]
