from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from musicaiz.structure import (
    Note,
    NoteClassNames,
    NoteClassBase
)


def get_highest_lowest_pitches(notes: List[Note]) -> Tuple[int, int]:
    """
    Extracts the highest and lowest pitches from a list of notes.

    Parameters
    ----------

    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.

    Returns
    -------

    highest_pitch, lowest_pitch: Tuple[int, int]
        The highest and lowest pitches.
    """
    lowest_pitch = 10000
    highest_pitch = 0
    for note in notes:
        if note.pitch < lowest_pitch:
            lowest_pitch = note.pitch
        if note.pitch > highest_pitch:
            highest_pitch = note.pitch
    return highest_pitch, lowest_pitch


def get_pitch_range(notes: List[Note]) -> int:
    """
    Computes the difference between the highest and the lowest pitches
    in a list of notes.

    Parameters
    ----------
    
    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.

    Returns
    -------
    
    int
        The pitch range.
    """
    high, low = get_highest_lowest_pitches(notes)
    return high - low


def _note_classes(notes: List[Note], feature: str) -> Dict[str, int]:
    """Counts the total number of different pitch classes.
    Based on:
    http://jmir.sourceforge.net/manuals/jSymbolic_manual/featureexplanations_files/featureexplanations.html

    Parameters
    ----------

    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.
    
    feature: str
        The feature used to count the notes which feature is the same.
        Valid Features: ``pitch``, ``note_name``, ``note_length``. 

    Returns
    -------

    notes_dict: Dict[str, int]. Format: {"pitch_value": counts}
        A dict with the feature as a key and the counts as the value.
    """
    if feature not in ["pitch", "note_name", "note_length"]:
        raise ValueError(f"{feature} is a non valid feature.")

    notes_dict = {}
    for note in notes:
        if feature == "pitch":
            attr = note.pitch
        elif feature == "note_name":
            attr = note.note_name
        elif feature == "note_length":
            attr = note.symbolic

        if str(attr) not in notes_dict.keys():
            notes_dict[str(attr)] = 1
        else:
            notes_dict[str(attr)] += 1
    return notes_dict


def get_note_density(notes: List[Note]) -> int:
    """Counts the total number of onsets in the ``Instrument``.
    
    Parameters
    ----------

    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.
    
    Returns
    -------
    
    int
        Note onset counts.
    """
    return len(notes)


def get_pitch_classes(notes: List[Note]) -> Dict[str, int]:
    """Counts the total number of different note classes
    :func:`~musicaiz.structure.NoteClassBase`.
    
    Parameters
    ----------

    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.

    Returns
    -------

    pitch_classes: Dict[str, int]. Format: {"pitch": counts}
        A dict with the feature as a key and the counts as the value.
    """
    pitch_classes = _note_classes(notes, feature="pitch")
    return pitch_classes


def get_note_classes(notes: List[Note]) -> Dict[str, int]:
    """Counts the total number of different note classes
    :func:`~musicaiz.structure.NoteClassBase`.
    
    Parameters
    ----------

    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.

    Returns
    -------

    note_classes: Dict[str, int]. Format: {"note_name": counts}
        A dict with the feature as a key and the counts as the value.
    """
    note_classes = _note_classes(notes, feature="note_name")
    return note_classes


def pitch_counts(notes: List[Note]) -> int:
    """Calculates the number of different pitches or Pitch Counts (PC)
    of a list of Notes.
    
    Parameters
    ----------

    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.
    
    Return
    ------

    int
        The number of different pitches in a list of :func:`~musicaiz.structure.Note` objects.
    """
    pitches = []
    for note in notes:
        if note.pitch not in pitches:
            pitches.append(note.pitch)
    return len(pitches)


def average_pitch_interval(notes: List[Note]) -> float:
    """
    Computes the Average Pitch Interval (PI) of a a list of `musicaiz`
    :func:`~musicaiz.structure.Note` objects.
    
    The steps to compute the PI are:
        1. Compute the pitch difference between two consecutive notes.
        2. Compute the mean between the pitch intervals.

    This method does take into account the polyphonic notes.

    Parameters
    ----------

    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.
    
    Returns
    -------

    float: _description_
        Average of the pitch intervals between consecutive notes.
    """
    # TODO: Ensure that notes are sorted by start_ticks.
    intervals = []
    for i in range(len(notes)):
        if i + 1 == len(notes):
            break
        intervals.append(abs(notes[i].pitch - notes[i + 1].pitch))
    if len(intervals) != 0:
        return sum(intervals) / len(intervals)
    else:
        return 0


def pitch_class_histogram(notes: List[Note]) -> np.ndarray:
    """
    Computes the Pitch Class Histogram (PCH) of a list of `musicaiz`
    :func:`~musicaiz.structure.Note` objects.

    Each element in the PCH array represents the counts of each note name in the
    chromatic scale of 12 notes.

    Parameters
    ----------

    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.

    Returns
    -------

    pitch_class_hist: np.ndarray [shape=(12,)]
        The PCH array.
    """
    note_classes = get_note_classes(notes)
    pitch_class_hist = np.zeros(12, dtype=int)
    for k, v in note_classes.items():
        note_class_name = NoteClassNames.get_note_with_name(k).name
        note_pos = NoteClassBase[note_class_name].chromatic_scale_index
        pitch_class_hist[note_pos] = v
    return pitch_class_hist


def _build_class_transition_dict(
    notes: List[Note],
    feature: str = "pitch"
) -> Dict[str, int]:
    """
    Builds a dict of the counts of each transition of a list of `musicaiz`
    :func:`~musicaiz.structure.Note` objects.
    
    Parameters
    ----------

    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.

    feature: str
        The feature used to count the notes which feature is the same.
        Valid Features: ``pitch``, ``note_length``. 

    Returns
    -------

    notes_dict: Dict[str, int]. Format: {feature: counts}
        A dict with the feature as a key and the counts as the value.
    """
    pitch_trans_dict = {}
    for i, note in enumerate(notes):
        if i != 0:
            if feature == "pitch":
                act_note = notes[i].note_name
                prev_note = notes[i - 1].note_name
            elif feature == "note_length":
                act_note = notes[i].symbolic.lower()
                prev_note = notes[i - 1].symbolic.lower()
            else:
                raise ValueError("feature must be `pitch` or `note_length`.")
            key = prev_note + "-" + act_note
            if key in pitch_trans_dict.keys():
                pitch_trans_dict[key] += 1
            else:
                pitch_trans_dict[key] = 1
    return pitch_trans_dict


def pitch_class_transition_matrix(notes: List[Note]) -> np.ndarray:
    """
    Computes the Pitch Class Transition Matrix (PCTM) of a list of `musicaiz`
    :func:`~musicaiz.structure.Note` objects.

    Parameters
    ----------

    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.

    Returns
    -------

    pctm: np.ndarray [shape=(12, 12)]
        The PCTM array.
    """
    pitch_trans_dict = _build_class_transition_dict(notes, feature="pitch")
    pctm = np.zeros(shape=(12, 12))
    for key, val in pitch_trans_dict.items():
        prev_note, next_note = key.split("-")[0], key.split("-")[1]
        prev_note_class_name = NoteClassNames.get_note_with_name(prev_note).name
        next_note_class_name = NoteClassNames.get_note_with_name(next_note).name
        prev_note_pos = NoteClassBase[prev_note_class_name].chromatic_scale_index
        next_note_pos = NoteClassBase[next_note_class_name].chromatic_scale_index
        pctm[prev_note_pos, next_note_pos] = val
    return pctm


def _plot_transition_matrix(matrix: np.array, labels: List[str]):
    """
    Plots a 2D np.ndarray with its list of labels.

    Parameters
    ----------
    
    matrix: np.ndarray [shape=(n,n)]
    
    labels: List[str]
    """
    fig, ax = plt.subplots(dpi=300)
    sns.heatmap(matrix, ax=ax, cmap="YlGnBu")
    ax.tick_params(length=0)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_aspect("equal")
    plt.show()


def plot_pitch_class_transition_matrix(pctm: np.array):
    """
    Plots the Pitch Class Transition Matrix (PCTM).

    Parameters
    ----------
    
    pctm: np.array [shape=(12,12)]
        The PCTM array.
    """
    labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    _plot_transition_matrix(pctm, labels)


def get_last_note_class(notes: List[Note]) -> int:
    """Returns the latest note name (`NoteClassBase`) in the sequence."""
    pass
