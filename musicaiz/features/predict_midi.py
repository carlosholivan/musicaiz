# This module uses the functions defined in the
# other `features` submodules to predict features
# from a midi files.


from typing import Union, TextIO, List


from musicaiz.loaders import Musa
from musicaiz.rhythm import get_subdivisions
from .rhythm import (
    get_start_sec,
    get_ioi,
    get_labeled_beat_vector,
    compute_all_rmss
)
from musicaiz.structure import Note


def _concatenate_notes_from_different_files(
    files: List[Union[str, TextIO]]
) -> List[Note]:
    # load midi file
    file_notes = []
    for i, file in enumerate(files):
        midi_object = Musa(file, structure="instruments")
        # extract notes from all instruments that are not drums
        file_notes += [instr.notes for instr in midi_object.instruments if not instr.is_drum]
    # Concatenate all lists into one
    all_notes = sum(file_notes, [])

    # extract subdivisions
    subdivisions = get_subdivisions(
        total_bars=midi_object.total_bars,
        subdivision="eight",
        time_sig=midi_object.time_sig,
    )
    return all_notes, subdivisions


def predic_time_sig_numerator(files: List[Union[str, TextIO]]):
    """Uses `features.rhythm` functions."""
    # load midi files and get all notes
    all_notes, subdivisions = _concatenate_notes_from_different_files(files)
    # 1. Get iois
    note_ons = get_start_sec(all_notes)
    iois = get_ioi(note_ons)
    # 2. Get labeled beat vector
    ioi_prime = get_labeled_beat_vector(iois)
    # 3. Get all rmss matrices
    all_rmss = compute_all_rmss(ioi_prime)
    # 4. Select rmss with more bar repeated instances
    # TODO
    pass
    return all_rmss
