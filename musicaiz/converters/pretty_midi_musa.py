from typing import Tuple
import pretty_midi as pm


from musicaiz.structure import NoteClassBase


def prettymidi_note_to_musicaiz(note: str) -> Tuple[str, int]:
    octave = int("".join(filter(str.isdigit, note)))
    # Get the note name without the octave
    note_name = note.replace(str(octave), "")
    musa_note_name = NoteClassBase.get_note_with_name(note_name)
    return musa_note_name.name, octave


def musicaiz_note_to_prettymidi(
    note: str,
    octave: int
) -> str:
    """
    >>> note = "F_SHARP"
    >>> octave = 3
    >>> pm_note = musicaiz_note_to_prettymidi(note, octave)
    >>> "F#3"
    """
    note_name = note.replace("SHARP", "#")
    note_name = note_name.replace("FLAT", "b")
    note_name = note_name.replace("_", "")
    pm_note = note_name + str(octave)
    return pm_note


def musa_to_prettymidi(musa_obj):
    """
    Converts a Musa object into a PrettMIDI object.

    Returns
    -------

    midi: PrettyMIDI
        The pretty_midi object.
    """
    # TODO: Write also metadata in PrettyMIDI object: pitch bends..
    midi = pm.PrettyMIDI(
        resolution=musa_obj.resolution,
        initial_tempo=musa_obj.tempo_changes[0]["tempo"]
    )
    midi.time_signature_changes = []
    for ts in musa_obj.time_signature_changes:
        midi.time_signature_changes.append(
            pm.TimeSignature(
                numerator=ts["time_sig"].num,
                denominator=ts["time_sig"].denom,
                time=ts["ms"] / 1000
            )
        )
    # TODO: Get ticks for each event (see Mido)
    midi._tick_scales = [
        (0, 60.0 / (musa_obj.tempo_changes[0]["tempo"] * midi.resolution))
    ]

    for i, inst in enumerate(musa_obj.instruments):
        midi.instruments.append(
            pm.Instrument(
                program=inst.program,
                is_drum=inst.is_drum,
                name=inst.name
            )
        )
        notes = musa_obj.get_notes_in_bars(
            bar_start=0, bar_end=musa_obj.total_bars,
            program=int(inst.program)
        )
        for note in notes:
            midi.instruments[i].notes.append(
                pm.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start_sec,
                    end=note.end_sec
                )
            )
    return midi
