import pretty_midi as pm
import numpy as np
from typing import List, Dict


from musicaiz.structure import Note
from musicaiz.rhythm import NoteLengths
from musicaiz.harmony import Chord


def predict_chords(musa_obj):
    notes_beats = []
    for i in range(len(musa_obj.beats)):
        nts = musa_obj.get_notes_in_beat(i)
        nts = [n for n in nts if not n.is_drum]
        if nts is not None or len(nts) != 0:
            notes_beats.append(nts)
    notes_pitches_segments = [_notes_to_onehot(note) for note in notes_beats]
    # Convert chord labels to onehot
    chords_onehot = Chord.chords_to_onehot()
    # step 1: Compute the distance between all the chord vectors and the notes vectors
    all_dists = [compute_chord_notes_dist(chords_onehot, segment) for segment in notes_pitches_segments]
    # step 2: get chord candidates per step which distance is the lowest
    chord_segments = get_chords_candidates(all_dists)
    # step 3: clean chord candidates
    chords = get_chords(chord_segments, chords_onehot)
    return chords


def get_chords(
    chord_segments: List[List[str]],
    chords_onehot: Dict[str, List[int]],
) -> List[List[str]]:
    """
    Clean the predicted chords that are extracted with get_chords_candidates method
    by comparing each chord in a step with the chords in the previous and next steps.
    The ouput chords are the ones wich distances are the lowest.

    Parameters
    ----------

    chord_segments: List[List[str]]
        The chord candidates extracted with get_chords_candidates method.

    Returns
    -------

    chords: List[List[str]]
    """
    chords = []
    for i, _ in enumerate(chord_segments):
        cross_dists = {}
        for j, _ in enumerate(chord_segments[i]):
            if i == 0:
                for item in range(len(chord_segments[i + 1])):
                    dist = np.linalg.norm(np.array(chords_onehot[chord_segments[i][j]]) - np.array(chords_onehot[chord_segments[i+1][item]]))
                    cross_dists.update(
                        {
                            chord_segments[i][j] + " " + chord_segments[i+1][item]: dist
                        }
                    )
            if i != 0:
                for item in range(len(chord_segments[i - 1])):
                    dist = np.linalg.norm(np.array(chords_onehot[chord_segments[i][j]]) - np.array(chords_onehot[chord_segments[i-1][item]]))
                    cross_dists.update(
                        {
                            chord_segments[i][j] + " " + chord_segments[i-1][item]: dist
                        }
                    )
        #print("--------")
        #print(cross_dists)
        chords_list = [(i.split(" ")[0], cross_dists[i]) for i in cross_dists if cross_dists[i]==min(cross_dists.values())]
        chords_dict = {}
        chords_dict.update(chords_list)
        #print(chords_dict)
        # Diminish distances if in previous step there's one or more chords equal to the chords in the current step
        for chord, dist in chords_dict.items():
            if i != 0:
                prev_chords = [c for c in chords[i - 1]]
                tonics = [c.split("-")[0] for c in prev_chords]
                tonic = chord.split("-")[0]
                if chord not in prev_chords or tonic not in tonics:
                    chords_dict[chord] = dist + 0.5
        #print(chords_dict)
        new_chords_list = [i for i in chords_dict if chords_dict[i]==min(chords_dict.values())]
        #print(new_chords_list)
        chords.append(new_chords_list)
    # If a 7th chord is predicted at a time step and the same chord triad is at
    # the prev at next steps, we'll substitute the triad chord for the 7th chord
    #for step in chords:
    #    chord_names = "/".join(step)
    #    if "SEVENTH" in chord_names:
    return chords


def get_chords_candidates(dists: List[Dict[str, float]]) -> List[List[str]]:
    """
    Gets the chords with the minimum distance in a list of dictionaries
    where each element of the list is a step (beat) corresponding to the note
    vectors and the items are dicts with the chord names (key) and dists (val.)

    Parameters
    ----------

    dists: List[Dict[str, float]]
        The list of distances between chord and note vectors as dictionaries per step.

    Returns
    -------

    chord_segments: List[List[str]]
        A list with all the chords predicted per step.
    """
    chord_segments = []
    for dists_dict in dists:
        chord_segments.append([i for i in dists_dict if dists_dict[i]==min(dists_dict.values())])
    return chord_segments


def compute_chord_notes_dist(
    chords_onehot: Dict[str, List[int]],
    notes_onehot: Dict[str, List[int]],
) -> Dict[str, float]:
    """
    Compute the distance between each chord and a single notes vector.
    The outpput is given as a dictionary with the chord name (key) and the distance (val.).

    Parameters
    ----------

    chords_onehot: Dict[str, List[int]]

    notes_onehot: Dict[str, List[int]]

    Returns
    -------

    dists: Dict[str, float]
    """
    dists = {}
    for chord, chord_vec in chords_onehot.items():
        dist = np.linalg.norm(np.array(notes_onehot)-np.array(chord_vec))
        dists.update({chord: dist})
    return dists


def _notes_to_onehot(notes: List[Note]) -> List[int]:
    """
    Converts a list of notes into a list of 0s and 1s.
    The output list will have 12 elements corresponding to
    the notes in the chromatic scale from C to B.
    If the note C is in the input list, the index corresponding
    to that note in the output list will be 1, otherwise it'll be 0.

    Parameters
    ----------
        notes: List[Note])

    Returns
    -------
        pitches_onehot: List[int]
    """
    pitches = [pm.note_name_to_number(note.note_name + "-1") for note in notes]
    pitches = list(dict.fromkeys(pitches))
    pitches_onehot = [1 if i in pitches else 0 for i in range(0, 12)]
    return pitches_onehot
