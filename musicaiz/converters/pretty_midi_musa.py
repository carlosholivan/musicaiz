from typing import Tuple
import pretty_midi as pm

from musicaiz.loaders import Musa
from musicaiz.structure import NoteClassBase


def pretty_midi_note_to_musanalysis(note: str) -> Tuple[str, int]:
    octave = int("".join(filter(str.isdigit, note)))
    # Get the note name without the octave
    note_name = note.replace(str(octave), "")
    musa_note_name = NoteClassBase.get_note_with_name(note_name)
    return musa_note_name.name, octave


def musa_to_prettymidi(musa_obj: Musa):
    """
    Converts a Musa object into a PrettMIDI object.

    Returns
    -------

    midi: PrettyMIDI
        The pretty_midi object.
    """
    # TODO: Write also metadata in PrettyMIDI object: pitch bends...
    midi = pm.PrettyMIDI(
        resolution=musa_obj.resolution,
        initial_tempo=musa_obj.bpm
    )

    for i, inst in enumerate(musa_obj.instruments):
        midi.instruments.append(
            pm.Instrument(
                program=inst.program,
                is_drum=inst.is_drum,
                name=inst.name
            )
        )
        for note in inst.notes:
            midi.instruments[i].notes.append(
                pm.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start_sec,
                    end=note.end_sec
                )
            )
    return midi