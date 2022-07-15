# This module uses the functions defined in the
# other `features` submodules to predict features
# from a midi files.


from typing import Union, TextIO, List, Tuple, Dict
import os
import multiprocessing


from musicaiz.harmony import (
    DegreesRoman,
    Tonality,
    ModeConstructors
)
from musicaiz.loaders import Musa
from .harmony import (
    predict_scales_degrees,
    predict_chords,
    predict_possible_progressions
)
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


def predict_midi_chords(
    files: Union[List[Union[str, TextIO]], str, TextIO],
) -> List[List[Tuple[DegreesRoman, Tonality, ModeConstructors]]]:
    """This funciton uses the `predict_chords` function which
    predicts the possible chords o a notes list but in this case
    for a whole midi file.

    The argument of this function is one or more midi files (it might
    be the case that we have the instruments in different midi files and
    we want to take all of them into account for the prediction)."""

    # load midi files and get all notes
    if not isinstance(files, list):
        files = [files]
    all_notes, subdivisions = _concatenate_notes_from_different_files(files)

    # loop in time steps of 20 ticks
    scales_degrees = []
    all_chords = []
    all_notes_steps = []

    # ticks step corresponding to the subdivision note
    ticks_step = subdivisions[-1]["ticks"] - subdivisions[-2]["ticks"]

    pool = multiprocessing.Pool(processes=os.cpu_count())
    for i in range(0, subdivisions[-1]["ticks"], ticks_step):
        notes_time_step = []
        for note in all_notes:
            # if note has already finished it's not in the current subdivision
            if note.end_ticks < i or note.start_ticks > i + ticks_step:
                continue
            # calculate note_one inside the current subdivision
            if note.start_ticks <= i:
                note_start = i
            else:
                note_start = note.start_ticks
            # calculate note_off inside the current subdivision
            if note.end_ticks < i + ticks_step:
                note_end = note.end_ticks
            else:
                note_end = i + ticks_step
            duration = note_end - note_start
            if duration >= ticks_step * 0.2:  # IF WE QUANTIZE, 0.5
                notes_time_step.append(note)
        all_notes_steps.append(notes_time_step)
        # Start the thread
        scales = pool.apply_async(
            predict_scales_degrees,
            args=(notes_time_step,)
        )
        # scales = features.predict_scales_degrees(notes_time_step)
        chords = predict_chords(notes_time_step)
        scales_degrees.append(scales)
        all_chords.append(chords)
    scales_degrees = [p.get() for p in scales_degrees]
    return scales_degrees


def predict_midi_all_keys_degrees(
    files: List[Union[str, TextIO]]
) -> Dict[str, List[Union[None, DegreesRoman]]]:
    scales_degrees = predict_midi_chords(files)
    all_scales = predict_possible_progressions(scales_degrees)
    return all_scales


def get_str_progression_from_scale(
    scales: Dict[str, List[Union[None, DegreesRoman]]],
    scale: str
) -> str:
    str_degrees = ""
    for i, degree in enumerate(scales[scale]):
        if i != 0:
            if i % 8 == 0:  # there are 8th notes per bar 4/4
                str_degrees += "|"
            else:
                str_degrees += "-"
        if degree is None:
            str_degrees += "None"
        else:
            str_degrees += degree.major
    return str_degrees


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
