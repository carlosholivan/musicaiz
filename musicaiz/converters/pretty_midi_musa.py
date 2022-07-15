from typing import Tuple

from musicaiz.structure import NoteClassBase


def pretty_midi_note_to_musanalysis(note: str) -> Tuple[str, int]:
    octave = int("".join(filter(str.isdigit, note)))
    # Get the note name without the octave
    note_name = note.replace(str(octave), "")
    musa_note_name = NoteClassBase.get_note_with_name(note_name)
    return musa_note_name.name, octave