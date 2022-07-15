""""
This module contains the implementation of the paper:

[1] Roig, C., Tardón, L. J., Barbancho, I., & Barbancho, A. M. (2014).
Automatic melody composition based on a probabilistic model of music
style and harmonic rules. Knowledge-Based Systems, 71, 419-434.
http://dx.doi.org/10.1016/j.knosys.2014.08.018

The implementation follows the paper method to predict rhythmic patterns.
This module contains:
1. Tempo (or bpm) estimation
    - get IOIs with `get_ioi` method.
    - get error ej
2. Time signature estimation
    - get the labeled beat vector (or IOI') from IOIs with `get_labeled_beat_vector`
    - get the Bar Split Vectors (BSV) for each beat (k in the paper) with `get_split_bar_vector`.
        k goes from 2 to 12 which are the most-common time_sig numerators.
    - compute the RSSM with each BSV with `compute_rhythm_self_similarity_matrix`.
    - get the time_sig numerator which will be the RSSM with the highest repeated
        bar instances.
3. Rhythm extraction
4. Pitch contour extraction
"""

from typing import List, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


from musicaiz.structure import Note
from musicaiz.rhythm import (
    SymbolicNoteLengths,
    NoteLengths,
)
from musicaiz.features.pitch import (
    _plot_transition_matrix,
    _build_class_transition_dict,
    _note_classes,
)


def get_start_sec(notes: List[Note]) -> List[Note]:
    """Extracts the time start of the notes in a notes sequence.
    
    Parameters
    ----------

    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.
    
    """
    all_note_on = []
    for note in notes:
        all_note_on.append(note.start_sec)
    all_note_on.sort()
    return all_note_on


def _delete_duplicates(
    all_note_on: List[Note]
) -> List[Note]:
    clean_list = list(dict.fromkeys(all_note_on))
    return clean_list


def get_ioi(
    all_note_on: List[Note],
    delete_overlap: bool = True
) -> List[Note]:
    """get ioi of a list of time start values"""

    if delete_overlap:
        all_note_on = _delete_duplicates(all_note_on)
    iois = []
    for i in range(len(all_note_on)):
        if i == 0:
            continue
        difference = all_note_on[i] - all_note_on[i - 1]
        iois.append(difference)
    return iois


def get_labeled_beat_vector(
    iois: List[float]
) -> List[int]:

    """
    Convert IOI to the labeled beat vector (or IOI')

    Example:
        IOI = [0.5, 0.375, 0.125] IOI' = [4, 4, 4, 4, 3, 3, 3, 1] eq(2) paper

    Parameters
    ----------

    iois: List[float]
        the IOIs which is a vector of IOI values in seconds (floats).

    Returns
    -------
    labeled_beat_vector: List[int]
        the IOI' vector that contains the beats and its values.
    """
    # A whole note (1) has 8 8th notes
    aux = [int(element * 8) for element in iois]
    labeled_beat_vector = []
    for note in aux:
        for i in range(0, note, 1):
            labeled_beat_vector.append(int(note))
    return labeled_beat_vector


def _split_labeled_beat_vector(
    labeled_beat_vector: List[int],
    beat_value: int
) -> List[List[int]]:

    """
    This function computes the Bar Split Vector for a given beat length value.
    Splits the labeled beat vector in measures with the beat
    value that corresponds to the beat or tactus (time_sig numerator).

    Parameters
    ----------

    labeled_beat_vector: List[int]
        the IOI' vector that contains the beats and its values.

    beat_value: int
        the note length value corresponding to a beat (time_sig numerator).
        This value is the paper `k` parameter which goes from 2 to 12 (common values).

    Returns
    -------

    splitted_vector: List[List[int]]
        the Bar Split Vector (BSV).
    """
    # if k (the beat value) is higher than the labeled_beat_vector length,
    # we can't split it so we won't split it
    if len(labeled_beat_vector) < beat_value:
        return labeled_beat_vector

    # split the labeled beat vector
    splitted_vector = []
    for k in range(0, len(labeled_beat_vector), beat_value):
        split = labeled_beat_vector[k:k + beat_value]
        splitted_vector.append(split)
    return splitted_vector


def compute_rhythm_self_similarity_matrix(
    splitted_beat_vector: List[List[int]],
) -> np.array:
    """
    This function computes the Rhythm Self-Similarity Matrix (RMSS).

    Parameters
    ----------

    splitted_beat_vector: List[List[int]]
        the splitted IOI' vector in bars.

    Returns
    -------

    rmss: np.array
        the Rhythm Self-Similarity Matrix
    """

    rmss = np.zeros(shape=(len(splitted_beat_vector), len(splitted_beat_vector)))
    for i, val_i in enumerate(splitted_beat_vector):
        for j, val_j in enumerate(splitted_beat_vector):
            if val_i == val_j:
                rmss[i, j] = 0
            else:
                rmss[i, j] = 1
    return rmss


def plot_rmss(
    rmss: np.array,
    k: Optional[int] = None,
    save: bool = True,
    filename: str = "rmss",
):
    plt.title(f"RMSS k={k}")
    plt.imshow(rmss, origin="lower")
    if save:
        plt.savefig(filename + ".png")
    plt.show()


# TODO
def compute_all_rmss(
    labeled_beat_vector: List[int],
) -> List[np.array]:
    """
    This function computes all the RMSS for time_sig numerators
    (k) 2 to 12 and outputs the beat (k) which will be the predicted
    time_sig numerator.
    The beat (k) is the RMSS which has more repeated bar instances.
    """
    # build vector of possible time_sig numerators
    k_values = [i for i in range(2, 12, 1)]
    # split vector
    all_rmss = []
    for beat_value in k_values:
        split = _split_labeled_beat_vector(labeled_beat_vector, beat_value)
        rmss = compute_rhythm_self_similarity_matrix(split)
        all_rmss.append(rmss)
    return all_rmss


def get_symbolic_length_classes(notes: List[Note]) -> Dict[str, int]:
    """Counts the total number of different note (`NoteClassBase`) classes.
    
    Parameters
    ----------
    
    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.
    
    """
    note_classes = _note_classes(notes, feature="note_length")
    return note_classes


def note_length_histogram(notes: List[Note]) -> np.array:
    """Uses `get_note_classes` to build a 1D list vector of 12 dimensions
    in which each element represents the counts of each note name in the
    chromatic scale of 12 notes.
    
    Parameters
    ----------
    
    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.
    
    """
    note_classes = get_symbolic_length_classes(notes)
    # we take into account triplets here. If we didn't take them into account
    # they'll just be zeros at the end of the list
    all_lengths = list(NoteLengths.get_note_ticks_mapping(True))
    note_length_hist = np.zeros(len(all_lengths), dtype=int)
    for k, v in note_classes.items():
        note_pos = all_lengths.index(k)
        note_length_hist[note_pos] = v
    return note_length_hist


def note_length_transition_matrix(
    notes: List[Note],
) -> np.array:
    """Computes the Note Length Transition MAtrix (NLTM) of a list of Note objects.
    
    Parameters
    ----------
    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.
    
    Returns
    -------
    
    
    """
    note_length_dict = _build_class_transition_dict(notes, feature="note_length")
    note_lengths = NoteLengths.get_note_ticks_mapping(True)
    nltm = np.zeros(shape=(len(note_lengths.values()), len(note_lengths.values())))
    for key, val in note_length_dict.items():
        prev_note, next_note = key.split("-")[0].upper(), key.split("-")[1].upper()
        for i, note_length in enumerate(list(note_lengths.keys())):
            if prev_note == note_length:
                prev_note_pos = i
            if next_note == note_length:
                next_note_pos = i
        nltm[prev_note_pos, next_note_pos] = val
    return nltm


def plot_note_length_transition_matrix(nltm: np.array):
    """Take into account that if you generated a NLTM with triplets = False
    then the triplet argument in this function will also be False."""
    # In case the NLTM was populated without taking into acocount triplets,
    # we remove them from the plot labels to let the labels len = matrix shape
    labels_dict = list(SymbolicNoteLengths.__members__.values())
    labels = [d.value for d in labels_dict]
    if len(labels) != nltm.shape[0]:
        raise ValueError("Labels len is not equal to the NLTM shape.")
    rcParams['font.sans-serif'] = ['Segoe UI Symbol','simHei','Arial','sans-serif']
    _plot_transition_matrix(nltm, labels)
