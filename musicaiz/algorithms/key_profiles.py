from typing import Tuple, List, Dict, Union
import math
import operator
from enum import Enum

from musicaiz.features import (
    pitch_class_histogram,
)
from musicaiz.structure import Note, NoteClassBase
from musicaiz.harmony import Tonality
from musicaiz.rhythm import NoteLengths


ORDERED_CIRCLE_FIFTHS = [
    "C", "G", "D", "A", "E", "B", "F_SHARP", "C_SHARP", "G_SHARP", "D_SHARP", "A_SHARP", "F"
]

CIRCLE_FIFTHS = [
    "A", "D", "G", "C", "F", "A_SHARP", "D_SHARP", "G_SHARP", "C_SHARP", "F_SHARP", "B", "E"
]

TONICS_MAPPING = {
   "C": "F_SHARP",
   "F": "B",
   "A_SHARP": "E",
   "D_SHARP": "A",
   "G_SHARP": "D",
   "C_SHARP": "G",
   "F_SHARP": "C",
   "B": "F",
   "E": "A_SHARP",
   "A": "D_SHARP",
   "D": "G_SHARP",
   "G": "C_SHARP",
}


class KeyDetectionAlgorithms(Enum):
    KRUMHANSL_KESSLER = ["k-k", "krumhansl-kessler"]
    TEMPERLEY = ["t", "temperley"]
    SIGNATURE_FIFTHS = ["5ths", "fifths"]
    ALBRETCH_SHANAHAN = ["a-s", "albrecht-shanahan"]

    @classmethod
    def all_values(cls) -> List[str]:
        all = []
        for n in cls.__members__.values():
            for name in n.value:
                all.append(name)
        return all


class KrumhanslKessler:

    """
    Krumhansl-Schmuckler/Kessler weights for
    Krumhansl-Schmuckler key-profiles algorithm.
    """
    
    @property
    def major(self):
        return {
            "C": 6.35,
            "C_SHARP": 2.23,
            "D": 3.48,
            "D_SHARP": 2.33,
            "E": 4.38,
            "F": 4.09,
            "F_SHARP": 2.52,
            "G": 5.19,
            "G_SHARP": 2.39,
            "A": 3.66,
            "A_SHARP": 2.29,
            "B": 2.88,
        }
    
    @property
    def minor(self):
        return {
            "C": 6.33,
            "C_SHARP": 2.68,
            "D": 3.52,
            "D_SHARP": 5.38,
            "E": 2.60,
            "F": 3.53,
            "F_SHARP": 2.54,
            "G": 4.75,
            "G_SHARP": 3.98,
            "A": 2.69,
            "A_SHARP": 3.34,
            "B": 3.17
        }


class Temperley:

    """
    Temperley/Kostka/Payne weights for
    key-profiles algorithm.
    """
    
    @property
    def major(self):
        return {
            "C": 0.748,
            "C_SHARP": 0.060,
            "D": 0.488,
            "D_SHARP": 0.082,
            "E": 0.67,
            "F": 0.460,
            "F_SHARP": 0.096,
            "G": 0.715,
            "G_SHARP": 0.104,
            "A": 0.366,
            "A_SHARP": 0.057,
            "B": 0.400,
        }
    
    @property
    def minor(self):
        return {
            "C": 0.712,
            "C_SHARP": 0.084,
            "D": 0.474,
            "D_SHARP": 0.618,
            "E": 0.049,
            "F": 0.460,
            "F_SHARP": 0.105,
            "G": 0.747,
            "G_SHARP": 0.404,
            "A": 0.067,
            "A_SHARP": 0.133,
            "B": 0.330
        }


class AlbrechtShanahan:

    """
    Albrecht/Shanahan weights for
    key-profiles algorithm.
    """
    
    @property
    def major(self):
        return {
            "C": 0.238,
            "C_SHARP": 0.006,
            "D": 0.111,
            "D_SHARP": 0.006,
            "E": 0.137,
            "F": 0.094,
            "F_SHARP": 0.016,
            "G": 0.214,
            "G_SHARP": 0.009,
            "A": 0.080,
            "A_SHARP": 0.008,
            "B": 0.081,
        }
    
    @property
    def minor(self):
        return {
            "C": 0.220,
            "C_SHARP": 0.006,
            "D": 0.104,
            "D_SHARP": 0.123,
            "E": 0.019,
            "F": 0.103,
            "F_SHARP": 0.012,
            "G": 0.214,
            "G_SHARP": 0.062,
            "A": 0.022,
            "A_SHARP": 0.061,
            "B": 0.052
        }


def signature_fifths(notes: List[Note]):
    # 1. Get the Pitch Class Histogram
    pc = pitch_class_histogram(notes)
    notes_scale = NoteClassBase.get_notes_chromatic_scale()
    # convert the notes to the circle of 5ths notation
    for i, note in enumerate(notes_scale):
        if note.name not in CIRCLE_FIFTHS:
            enharmonics = NoteClassBase._get_note_from_chromatic_idx(note.chromatic_scale_index)
            if enharmonics[0].name in CIRCLE_FIFTHS:
                notes_scale[i] = enharmonics[0]
            else:
                notes_scale[i] = enharmonics[1]

    # 2. Get pitch multiplicity (k = x_i / max(x))
    hist = {}
    for i, note in enumerate(notes_scale):
        hist.update({note.name: pc[i] / max(pc)})

    notes_axes = {}
    for note in hist.keys():
        r, l = _right_left_notes(note)
        k_r = [hist[i] for i in r]
        k_l = [hist[i] for i in l]
        # this is different from the paper to correct the sign due to this implementation
        axis = sum(k_l) - sum(k_r)
        notes_axes.update({note: axis})
    return notes_axes


def _signature_fifths_keys(notes_axes) -> Tuple[str, float]:
    # 3. The right note in the ORDERED_CIRCLE_FIFTHS of the maximum directed axes
    # will be the possible major key and its relative minor key will be the other possible key

    # Order notes following the ORDERED_CIRCLE_FIFTHS
    notes_axes = sorted(notes_axes.items(), key=lambda x:ORDERED_CIRCLE_FIFTHS.index(x[0]))
    notes_axes = dict((x, y) for x, y in notes_axes)

    # What happens when we have equal maximum values?
    # We take the 1st one starting by the ORDERED_CIRCLE_FIFTHS
    for note, v in notes_axes.items():
        if v == max(notes_axes.values()):
            major_key_tonic = note
            break

    major_key_tonic_idx = ORDERED_CIRCLE_FIFTHS.index(major_key_tonic) + 1
    major_key_tonic = ORDERED_CIRCLE_FIFTHS[major_key_tonic_idx]
    # Find the minor key tonic as the tonality with the same number of # or b of the major key
    if major_key_tonic + "_MAJOR" not in [n for n in Tonality.__members__]:
        # look for enharmonic note to construct a correct tonality
        chr = NoteClassBase[major_key_tonic].chromatic_scale_index
        notes = NoteClassBase._get_note_from_chromatic_idx(chr)
        notes = [n.name for n in notes]
        tonic = set(notes).difference(set([major_key_tonic]))
        (new_major_key_tonic,) = tonic
        minor_key = Tonality[new_major_key_tonic + "_MAJOR"].relative.name
    else:
        minor_key = Tonality[major_key_tonic + "_MAJOR"].relative.name
    # We calculate the correlation coeficients "r" for both keys, the maximum will denote the key

    # Construct the profiles dict that is a reduction of the notes_Axes with only the possible tonalities
    major = (major_key_tonic + "_MAJOR", notes_axes[major_key_tonic])
    minor = (minor_key, notes_axes[minor_key.split("_MINOR")[0]])
    return major, minor

# Given a note, compute right and left multiplicity differences
def _right_left_notes(note: str) -> Tuple[List[str], List[str]]:
    tonic_idx = CIRCLE_FIFTHS.index(note)
    double_circle_fifths = [*CIRCLE_FIFTHS, *CIRCLE_FIFTHS]
    right = double_circle_fifths[tonic_idx+1:tonic_idx+6]
    left = double_circle_fifths[tonic_idx+7:tonic_idx+7+5]
    return right, left


def _correlation(
    keys: Dict[str, Dict[str, Union[int, float]]]
) -> Dict[str, float]:
    corr_keys = {}
    for key, val in keys.items():
        x_y, x_2, y_2 = [], [], []
        pitch = [v[0] for v in val.values()]
        dur = [v[1] for v in val.values()]
        for v in val.values():
            x_diff = v[0] - (sum(pitch) / len(pitch))
            y_diff = v[1] - (sum(dur) / len(dur))
            x_y.append(x_diff * y_diff)
            x_2.append(x_diff**2)
            y_2.append(y_diff**2)
        corr_keys.update({key: (sum(x_y) / math.sqrt(sum(x_2) * sum(y_2)))})
    return corr_keys


def _keys_correlations(
    durations: Dict[str, int],
    method: str
) -> Dict[str, float]:
    # Construct pitch-profile values for each key
    keys = {}
    for note, _ in durations.items():
        major_key, minor_key = {}, {}
        note_idx = list(durations.keys()).index(note)
        values = list(durations.values())
        values = values[note_idx:] + values[:note_idx]
        for i, l_note in enumerate(durations.keys()):
            if method == "k-k":
                major_key.update({l_note: (KrumhanslKessler().major[l_note], values[i])})
                minor_key.update({l_note: (KrumhanslKessler().minor[l_note], values[i])})
            elif method == "temperley":
                major_key.update({l_note: (Temperley().major[l_note], values[i])})
                minor_key.update({l_note: (Temperley().minor[l_note], values[i])})
            elif method == "a-s":
                major_key.update({l_note: (AlbrechtShanahan().major[l_note], values[i])})
                minor_key.update({l_note: (AlbrechtShanahan().minor[l_note], values[i])})
        keys.update({note + "_MAJOR": major_key})
        keys.update({note + "_MINOR": minor_key})

    corr_keys = _correlation(keys)
    return corr_keys


def signature_fifths_profiles(
    notes_axes: Dict[str, float]
) -> Dict[str, float]:
    # Get the possible keys
    major, minor = _signature_fifths_keys(notes_axes)
    
    keys = {}
    major_key, minor_key = major[0], minor[0]
    major_key_tonic, minor_key_tonic = major_key.split("_MAJOR")[0], minor_key.split("_MINOR")[0]
    major_note_idx = list(notes_axes.keys()).index(major_key_tonic)
    minor_note_idx = list(notes_axes.keys()).index(minor_key_tonic)
    major_values = list(notes_axes.values())
    major_values = major_values[major_note_idx:] + major_values[:major_note_idx]
    minor_values = list(notes_axes.values())
    minor_values = minor_values[minor_note_idx:] + minor_values[:minor_note_idx]
    maj, mi = {}, {}
    for i, l_note in enumerate(notes_axes.keys()):
        maj.update({l_note: (KrumhanslKessler().major[l_note], major_values[i])})
        mi.update({l_note: (KrumhanslKessler().minor[l_note], minor_values[i])})
    keys.update({major_key: maj})
    keys.update({minor_key: mi})

    corr_keys = _correlation(keys)
    return corr_keys


def _eights_per_pitch_class(notes: List[Note]) -> Dict[str, int]:
    """Computes the number of eight notes per pitch class"""
    pitches_ticks = {} # A dict with the duration ticks of each
    # pitch class: Dict[str, int] {"C": 245...}
    for note in notes:
        dur = note.end_ticks - note.start_ticks
        if note.note_name not in pitches_ticks.keys():
            pitches_ticks.update({note.note_name: dur})
        else:
            pitches_ticks[note.note_name] += dur

    # Divide the total ticks per pitch class by the ticks that corr. to 1 8th note
    ticks_eight = NoteLengths.EIGHT.ticks()
    durations = {}
    for pitch, v in pitches_ticks.items():
        durations.update({pitch: int(v / ticks_eight)})
    
    # Add the pitch classes that are not present in the passed notes
    dur = {}
    for note in durations.keys():
        # TODO: Possible bug if the pitch class is a flat and not # 
        new_note = note.replace("#", "_SHARP")
        new_note = new_note.replace("b", "_FLAT")
        dur.update({new_note: durations[note]})
    for n in ORDERED_CIRCLE_FIFTHS:
        if n not in dur.keys():
            dur.update({n: 0})
    pitches = NoteClassBase.get_notes_chromatic_scale("SHARP")
    pitches = [p.name for p in pitches]
    dur = sorted(dur.items(), key=lambda x:pitches.index(x[0]))
    dur = dict((x, y) for x, y in dur)
    return dur


def key_detection(notes: List[Note], method: str = "k-k"):
    """
    Algorithm description:

    1. Count the number of eights of each pitch class.
    2. Store the duration values in the `durations` dict (input arg. of this function).
    3. Calculate the correlation between pitch weights and the durations
    (this in done with :func:`~musicaiz.algorithms._keys_correlations`.)
    4. The higher correlation is corresponds to the key.

    Parameters
    ----------

    durations (Dict[str, int]): _description_
        method (str, optional): _description_. Defaults to "k-k".

    Returns
    -------
    
    _type_: _description_
    """
    if method not in KeyDetectionAlgorithms.all_values():
        raise ValueError("Not key detection algorithm found.")

    durations = _eights_per_pitch_class(notes)
    if method in KeyDetectionAlgorithms.KRUMHANSL_KESSLER.value:
        corr = _keys_correlations(durations, "k-k")
    elif method in KeyDetectionAlgorithms.TEMPERLEY.value:
        corr = _keys_correlations(durations, "temperley")
    elif method in KeyDetectionAlgorithms.ALBRETCH_SHANAHAN.value:
        corr = _keys_correlations(durations, "a-s")
    elif method in KeyDetectionAlgorithms.SIGNATURE_FIFTHS.value:
        notes_axes = signature_fifths(notes)
        corr = signature_fifths_profiles(notes_axes)
    return max(corr.items(), key=operator.itemgetter(1))[0]
