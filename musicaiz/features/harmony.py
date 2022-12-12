from typing import List, Tuple, Dict, Union, Optional
import warnings
import itertools
import copy


from musicaiz.harmony import (
    AllChords,
    Interval,
    IntervalSemitones,
    ChordType,
    Scales,
    Tonality,
    DegreesRoman,
    ModeConstructors,
)
from musicaiz.structure import (
    Note,
    NoteClassBase,
)


def _extract_note_positions(note_seq: List[Note]) -> List[int]:
    """
    Extracts the note positions in the chromatic scale of the notes in a notes sequence.

    Parameters
    ----------

    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.

    Returns
    -------

    notes_position: List[int]
        A list of the position in the chromatic scale of each note in the input notes list.
    """
    notes_position = []
    for note in note_seq:
        note_obj = NoteClassBase.get_note_with_name(note.note_name)
        position = note_obj.chromatic_scale_index
        notes_position.append(position)
    return notes_position


def _order_note_seq_by_chromatic_idx(note_seq: List[Note]) -> List[int]:
    """
    Sorts a note seq (list of note objects) by the index of the notes
    in the chromatic scale.

    Parameters
    ----------

    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.

    Returns
    -------

    sorted_note_seq: List[int]
        Sorts the positions of the notes in the chromatic scale.

    """
    # Extract the notes indexes in the natural scale
    positions = _extract_note_positions(note_seq)
    # Sorts the list of indexes
    sorted_positions = sorted(positions)
    # Sorts the note seq list
    sorted_note_seq = [None] * len(note_seq)
    for idx, item in enumerate(sorted_positions):
        note_seq_pos = positions.index(item)
        sorted_note_seq[idx] = note_seq[note_seq_pos]
    return sorted_note_seq


def get_chord_type_from_note_seq(note_seq: List[Note]) -> str:
    """
    Gets the chord type :func:`~musicaiz.harmony.ChordType` of a list of
    `musicaiz` :func:`~musicaiz.structure.Note` objects.

    The chord type is equal to the number of note names or different notes positions
    in the chromatic scale that are different in the note seq.
    This functions return the minimum chord type of a note seq.
    3 different note names are a triad, 4 notes a 7th or 9th (7th the minimum),
    5 notes a 9th with the 7th and so on.

    Parameters
    ----------
    
    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.

    Returns
    -------

    chord_type: str
        The type of the chord :func:`~musicaiz.harmony.ChordType`.
    """
    notes_positions = _extract_note_positions(note_seq)
    # count the number of different elements in the notes_positions list
    min_type = len(set(notes_positions))
    chord_type = ChordType.get_type_from_value(min_type)
    return chord_type


def get_intervals_note_seq(note_seq: List[Note]) -> List[List[IntervalSemitones]]:
    """
    Get the intervals between pairs of notes (1st note and the rest of
    the notes in the note seq) of a sorted note seq.
    This computes all the intervals between 2 notes, taking into account the
    notes enharmonics.

    Parameters
    ----------

    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.

    """
    intervals = []
    for i in range(1, len(note_seq)):
        interval = Interval.get_possible_intervals(note_seq[0], note_seq[i])
        intervals.append(interval)
    return intervals


def predict_chords(
    note_seq: List[Note]
) -> List[Tuple[NoteClassBase, AllChords]]:
    """Predicts a chord in a note sequence with only note values,
    so no note durations atr taken into account.

    Parameters
    ----------

    notes: List[Note]
        A list of `musicaiz` :func:`~musicaiz.structure.Note` objects.

    """
    # Check note seq length
    if len(note_seq) <= 1:
        return []
    # Remove duplicated notes in the note seq (12 and 24 are C, we delete 24)
    note_seq = _delete_repeated_note_names(note_seq)
    # TODO: Compute note seq with enharmonics
    # We generate all possible note seq permutations bc we don't know what
    # is the root note of the chord nor the inversion of the chord
    all_note_seqs = _all_note_seq_permutations(note_seq)
    predicted_chords = []
    for single_note_seq in all_note_seqs:
        # 1. Measure intervals between pairs of notes
        # intervals vary if we take enharmonic notes(C - Eb is not equal to C - D#)
        # intervals is a list of lists of intervals between the enharmonics pairs of notes
        # so we need to map this to all the possible combinations of intervals
        all_intervals_tuples = get_intervals_note_seq(single_note_seq)
        all_possible_tuples = list(itertools.product(*all_intervals_tuples))
        # Remove tuple of (notes, intervals) to only have intervals as items
        # TODO: Refactor this in other method
        all_intervals = []
        for a_tuple in all_possible_tuples:
            all_intervals.append([tup[1] for tup in a_tuple])
        # 2. Look into all the possible with a type equal or higher than the note seq type
        for intervals in all_intervals:
            for chord in AllChords:
                # Now check if the list of intervals corresponds to a chord
                # We might have intervals that are not in the defined chords due to passing notes
                # so we check if the chord intervals set is a subset of the predicted intervals
                note = NoteClassBase.get_note_with_name(single_note_seq[0].note_name)
                chord_tuple = (note, chord)
                if set(intervals) >= set(chord.value[1]) and chord_tuple not in predicted_chords:
                    predicted_chords.append(chord_tuple)
    if len(predicted_chords) == 0:
        warnings.warn("No chord has been matched.")
    # We store in predicted_chord all the chords that are subsets of out list of predicted
    # intervals but the last chord in the list will be the more complex chord detected in
    # (triads will be save 1st, then 7ths, then 9ths...if they match the predicted intervals)
    # TODO: Return also the inversion
    # TODO: What if the chord has no all its notes?
    # If 2 chords have the same root note remove the one with less complexity (if triad and 7th are found, remove triad)
    # This is bc triads are subsets of 7th which are subsets of 9ths...and we've stored them all
    chords = copy.deepcopy(predicted_chords)
    for pred in predicted_chords:
        for pred1 in predicted_chords[1:]:
            if pred[0] == pred1[0]:  # root notes are the same
                if pred[1].chord_type.value < pred1[1].chord_type.value:
                    chords.remove(pred)
                    break

    # Now give also the enharmonic chords to the obtained chords
    # Ex.: C major is the same as B# major if we look at note's pitches
    # TODO: Refactor this in other method
    all_chords = copy.deepcopy(chords)
    for ch in chords:
        root_note = ch[0]
        degree_root = root_note.value[0].contracted + "1"
        note_obj = Interval._initialize_note(degree_root)
        enharmonic_root_note = note_obj.enharmonic
        # Some notes (D...) have no enharmonics (only considering notes with 1 accidental)
        if enharmonic_root_note != root_note:
            all_chords.append((enharmonic_root_note, ch[1]))
    return all_chords


def predict_scales_degrees(
    note_seq: List[Note]
) -> List[List[Tuple[DegreesRoman, Tonality, ModeConstructors]]]:
    """This method is similar to `Scales.get_scales_degrees_from_chord`
    method but in this case applied to an input note_seq, not to a chord.
    So what we do here is applying chord detection to the note_Seq and then
    the sacle and degrees mapping for every detected chord in the note_seq."""
    possible_chords = predict_chords(note_seq)
    all_scales_degrees = []
    for chord in possible_chords:
        scales_degrees = Tonality.get_scales_degrees_from_chord(chord)
        all_scales_degrees.extend(scales_degrees)
    return all_scales_degrees


def predict_possible_progressions(
    possible_chords: List[List[Tuple[NoteClassBase, AllChords, ModeConstructors]]],
    scale: Optional[Union[str, Scales]] = None,
) -> Dict[str, List[DegreesRoman]]:
    """
    Get all possible scales and degrees from a chords list (chord progression)
    We retrieve a list of degrees which items correspond to one time step each.

    Parameters
    ----------

    possible_chords: list
        the possible chords in each time step. This argument is a list, in which
        each element is a time step with is another list of the possible degrees that
        can belong to the same time step. The chords are represented by a tuple of
        (`degree`, `tonality` and `mode`).

    scale: str or Scale object.
        if we do know the scale, the funcion will return only the dict with the key of
        the input scale and the degrees that belong to that scale. Otherwise we'll predict
        all the possible scales and map each chord to its possible scales.

    Returns
    -------
    """
    # Go through all the chords in each time step or subdivision
    # Create dict with keys equal to all existing scales in the preogression
    scales = {}
    for chords_step in possible_chords:
        if len(chords_step) != 0:
            for chords in chords_step:
                scale_name = chords[1].name
                if scale_name not in scales.keys():
                    new_scale = {scale_name: []}
                    scales = {**scales, **new_scale}

    # Fill the degrees list for every scales with the degree value in each time step
    for chords_step in possible_chords:
        if len(chords_step) != 0:
            scales_time_step = []
            step_degrees_scales = []
            for i, chords in enumerate(chords_step):
                degree_name = chords[0]
                scale_name = chords[1].name
                scales_time_step.append(scale_name)
                # if degree of a certain scale has been already written in a time step,
                # we don't append it to not have duplicates
                if (degree_name, scale_name) not in set(step_degrees_scales):
                    scales[scale_name].append(degree_name)
                step_degrees_scales.append((degree_name, scale_name))
            scales_not_in_step = set(scales.keys()) - set(scales_time_step)
            for sc in scales_not_in_step:
                scales[sc].append(None)
        else:
            for scale in scales.keys():
                scales[scale].append(None)
    return scales


def predict_progression(
    scales: Dict[str, List[DegreesRoman]],
    scale: Optional[Union[str, Scales]],
) -> Tuple[str, List[DegreesRoman]]:
    """Uses `predict_possible_progressions` to predict all the scales and progressions
    that belong to a note_seq but this method only returns one of them.
    If the scale is  known, it returns the progression of that scale, otherwise
    this method will return the scale that has more known degrees in the progression.
    Ex.: If a scale has only one known degree in all the time steps, it is probable that
    the scale is not the correct one (although it is a secundary dominant. If the scale
    has a predicted degree more time steps the probability of that scale being the
    correct one increases."""
    if scale is not None:
        if isinstance(scale, str):
            return scale, scales[scale]
        else:
            scale_name = scale.name
            return scale_name, scales[scale_name]
    else:
        # TODO: Return scale with less None values
        return scales[scale]


def _all_note_seq_permutations(note_seq: List[Note]) -> List[List[Note]]:
    """Returns a list of lists of all possible orders for a note seq."""
    return [list(p) for p in itertools.permutations(note_seq)]


def _delete_repeated_note_names(note_seq: List[Note]) -> List[Note]:
    """Removes a repeated note in a note_seq.
    A repeated note is a note which name (`C`) not pitch is equal to other note name."""
    note_names = []
    new_note_seq = []
    for note in note_seq:
        if note.note_name not in note_names:
            new_note_seq.append(note)
        note_names.append(note.note_name)
    return new_note_seq


def get_harmonic_density(note_seq: List[Note]) -> int:
    """Computes the maximum number of notes that are overlapped in the
    harmonic axis."""
    # No notes in the sequence means that density is 0.
    if len(note_seq) == 0:
        return 0
    # Go tick per tick
    latest_note = note_seq[-1]
    counts = []
    step_ticks = 10
    # We'll compute by steps of 10 ticks which is a low value
    for i in range(0, latest_note.end_ticks, step_ticks):
        count = 0
        for note_idx, note in enumerate(note_seq):
            # if note ends aftre the next step start, count it
            if note.start_ticks < i and note.end_ticks >= i:
                count += 1
        counts.append(count)
    return max(counts)
