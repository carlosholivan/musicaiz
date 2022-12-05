import pretty_midi as pm
from typing import List


from musicaiz.harmony import Tonality
from musicaiz.structure import NoteClassBase, Note
from musicaiz.converters import prettymidi_note_to_musicaiz


def harmonic_shifting(
    origin_notes: List[List[Note]],
    origin_progression: List[List[str]],
    origin_tonality: str,
    origin_scale: str,
    target_progression: List[List[str]],
    target_tonality: str,
    target_scale: str,
) -> List[List[Note]]:

    """
    This function maps the pitches of a midi to a new scale and chord progression.
    To do that, given the progression of the input midi and the target scale and
    progression, we analyze the note position in the chord built from a degree and
    we map the pitch to the same position in the target degree chord.
    Note that for using this function we should know in advance the degrees that
    are contained in each bar, but we don't need to know where are the transitions
    from a degree to another inside the bar.

    Parameters
    ----------
    origin_notes: List[Note]
        the original midi data.

    origin_progression: List[List[str]]
        the target degrees per bar.

    origin_tonality: str
        the origin tonality.

    origin_scale: str
        the origin scale.

    target_progression: List[List[str]]
        the target degrees per bar.

    target_tonality: str
        the target scale.

    target_scale: str
        the target scale.

    Returns
    -------
    origin_notes: List[Note]

    Examples
    --------
    Example of 2 bars, 1st bar with 3 notes and 2nd bar with 1 note:

    >>> origin_notes = [  # each list corresponds to the notes in a bar
    >>>     [
    >>>         Note(pitch=43, start=0, end=96, velocity=82),
    >>>         Note(pitch=58, start=96, end=96*2, velocity=82),
    >>>         Note(pitch=60, start=96*2, end=96*3, velocity=82),
    >>>     ],
    >>>     [
    >>>         Note(pitch=72, start=96*4, end=96*5, velocity=44)
    >>>     ]
    >>> ]
    >>> origin_bars = [["II", "IV"], ["VII"]]
    >>> origin_tonality = "G_MINOR"
    >>> origin_scale = "NATURAL"
    >>> target_progression = [["I", "V"], ["IV"]]
    >>> target_tonality = "C_MINOR"
    >>> target_scale = "NATURAL"
    """

    for bar_idx, bar in enumerate(origin_notes):
        note_idx = 0
        for idx, (orig_degree, target_degree) in enumerate(
            zip(origin_progression[bar_idx], target_progression[bar_idx])
        ):
            for note in bar[note_idx:]:
                new_pitch = 0
                # CHECK IF NOTE IS IN THE DEGREE
                orig_pitch = note.pitch
                # get the note name from pitch
                orig_note_name = pm.note_number_to_name(orig_pitch)
                # Extract octave
                octave = int("".join(filter(str.isdigit, orig_note_name)))
                # Get the note name without the octave
                orig_note_name = orig_note_name.replace(str(octave), "")
                # get notes from chord degree
                orig_note_names_str = _get_chord_note_names(
                    origin_tonality, origin_scale, orig_degree
                )
                # Get the note position in the chord of the degree
                note_position = [
                    i
                    for i, note in enumerate(orig_note_names_str)
                    if note == orig_note_name
                ]

                # if no note has been matched with the chord, we'll check if there's other degree in the bar
                # to see if the note can belong to the other degree. If the note does not belong to any
                # degree chord in the bar, it's a passing note
                if len(note_position) == 0:
                    # check if note belongs to the next degree
                    if len(origin_progression[bar_idx][idx + 1 :]) > 0:
                        next_degree_in_bar = origin_progression[bar_idx][idx + 1]
                        # get notes from chord degree
                        orig_note_names_str = _get_chord_note_names(
                            origin_tonality, origin_scale, next_degree_in_bar
                        )
                        note_position = [
                            i
                            for i, note in enumerate(orig_note_names_str)
                            if note in orig_note_name
                        ]
                        # if note is in the next degree break loop to go to next degree
                        if len(note_position) != 0:
                            # ==============NOTE IN NEXT DEGREE========
                            break
                        # if note is not in the next degree we assume it's a passing note
                        else:
                            # ==============PASSING NOTE===============
                            target_pitch = _map_passing_note(
                                orig_pitch,
                                origin_tonality,
                                origin_scale,
                                orig_degree,
                                target_tonality,
                                target_scale,
                                target_degree,
                            )
                            if orig_pitch <= new_pitch:
                                note.pitch = target_pitch
                            else:
                                note.pitch = target_pitch + 12

                            # we already wrote the note pitch so we'll go to the next one
                            note_idx += 1
                            continue
                    # if there aren't more degrees in the bar to check we assume it's a passing note
                    else:
                        # ==============PASSING NOTE===============
                        target_pitch = _map_passing_note(
                            orig_pitch,
                            origin_tonality,
                            origin_scale,
                            orig_degree,
                            target_tonality,
                            target_scale,
                            target_degree,
                        )

                        if orig_pitch <= new_pitch:
                            note.pitch = target_pitch
                        else:
                            note.pitch = target_pitch + 12

                        # we already wrote the note pitch so we'll go to the next one
                        note_idx += 1
                        continue

                note_position = note_position[0]
                note_idx += 1

                # With this position, go to the destination degree and get the note
                dest_note_names_str = _get_chord_note_names(
                    target_tonality, target_scale, target_degree
                )
                new_note_name = dest_note_names_str[note_position]
                new_note = new_note_name + str(octave)
                # convert note into pitch
                new_pitch = pm.note_name_to_number(new_note)
                # substitute the old pitch by the new one
                # we sum an octave to the note if the original note is higher than the
                # new mapped note to perserve the pitch contour of the dataset execution's id
                if orig_pitch <= new_pitch:
                    note.pitch = new_pitch
                else:
                    note.pitch = new_pitch + 12

    return origin_notes


def scale_change(
    origin_notes: List[Note],
    origin_tonality: str,
    origin_scale: str,
    target_tonality: str,
    target_scale: str,
    correction: bool
) -> List[Note]:

    """
    This function maps the pitches of a midi to a new scale.
    To do that, the position of the note in the original scale is
    extracted and then it is used to obtain the pitch in the target
    scale.
    The pitches are mapped to the same octave than the original ones.

    Parameters
    ----------
    origin_notes: List[Note]
        the original midi data.

    origin_tonality: str
        the origin tonality.

    origin_scale: str
        the origin scale.

    target_tonality: str
        the target scale.

    target_scale: str
        the target scale.
    
    correction: bool
        if we want to correct the "wrong" input notes.
        If True it corrects the non belonging notes by adding a semitone to the
        input note that does not belong to the origin_scale.
        If False the algorithm does not correct the input notes. The note that does
        not belong to the input scale will be mapped by calculating the
        difference in semitones between the note and the tonic of the scale, and
        then the semitones are added to the tonic of the target scale to obtain the target note.

    Returns
    -------
    origin_notes: List[Note]

    Examples
    --------
    Example of 2 bars, 1st bar with 3 notes and 2nd bar with 1 note:

    >>> origin_notes = [
    >>>         Note(pitch=43, start=0, end=96, velocity=82), # G
    >>>         Note(pitch=58, start=96, end=96*2, velocity=82), # A#
    >>>         Note(pitch=60, start=96*2, end=96*3, velocity=82), # C
    >>>         Note(pitch=72, start=96*4, end=96*5, velocity=44) # C
    >>> ]
    >>> origin_tonality = "G_MINOR"
    >>> origin_scale = "NATURAL"
    >>> target_tonality = "C_MINOR"
    >>> target_scale = "NATURAL"
    """

    # Get the note position in the scale
    orig_tonality = Tonality[origin_tonality]
    orig_scale_notes = [note.name for note in orig_tonality.scale_notes(origin_scale)]

    target_tonality = Tonality[target_tonality]
    target_scale_notes = [note.name for note in target_tonality.scale_notes(target_scale)]

    for note in origin_notes:
        orig_note_name = pm.note_number_to_name(note.pitch)

        # Extract octave
        octave = int("".join(filter(str.isdigit, orig_note_name)))

        # Get the note name without the octave
        orig_note_name = orig_note_name.replace(str(octave), "")

        if orig_note_name in orig_scale_notes:
            note_position = orig_scale_notes.index(orig_note_name)
        else:
            if correction is True:
                # Get note 1 semitone up than the original
                orig_note_name = orig_note_name.replace("#", "_SHARP")
                orig_note_name = orig_note_name.replace("b", "_FLAT")
                orig_note_name = NoteClassBase[orig_note_name].add_sharp.name
                note_position = orig_scale_notes.index(orig_note_name)
            else:
                orig_tonic = orig_scale_notes[0]
                orig_tonic = orig_tonic.replace("_SHARP", "#")
                orig_tonic = orig_tonic.replace("_FLAT", "b")
                orig_tonic_pitch = pm.note_name_to_number(orig_tonic + str(octave))
                diff_semitones = abs(note.pitch - orig_tonic_pitch)

                target_tonic = target_scale_notes[0]
                target_tonic = target_tonic.replace("_SHARP", "#")
                target_tonic = target_tonic.replace("_FLAT", "b")
                target_tonic_pitch = pm.note_name_to_number(target_tonic + str(octave))
                target_pitch = diff_semitones + target_tonic_pitch
                target_note_name = pm.note_number_to_name(target_pitch)

                target_octave = int("".join(filter(str.isdigit, target_note_name)))
                if target_octave < octave:
                    target_pitch += 12
                elif target_octave > octave:
                    target_pitch -= 12
                new_note_name = target_note_name.replace(str(octave), "")
                new_note_name = new_note_name.replace("_SHARP", "#")
                new_note_name = new_note_name.replace("_FLAT", "b")

                note.pitch = target_pitch
                note.note_name = new_note_name
                note.pitch_name = target_note_name
                continue

        target_note_name = target_scale_notes[note_position]

        target_note_name = target_note_name.replace("_SHARP", "#")
        target_note_name = target_note_name.replace("_FLAT", "b")

        new_note_name = target_note_name + str(octave)
        new_pitch = pm.note_name_to_number(new_note_name)

        note.pitch = new_pitch
        note.note_name = target_note_name
        note.pitch_name = new_note_name
    return origin_notes


def _map_passing_note(
    origin_pitch: int,
    origin_tonality: str,
    origin_scale: str,
    origin_degree: str,
    target_tonality: str,
    target_scale: str,
    target_degree: str,
) -> int:
    """Map a passing note to a passing note in the new scale.
    For doing that, we measure the semitones distance between the
    note and the tonic of the 1st degree of the scale. Then, we take the
    tonic of the 1st degree of the target scale and we sum those semitones to it
    """

    # get the pitch of the tonic of the 1st degree
    orig_tonic_note_name_str = (
        _get_chord_note_names(origin_tonality, origin_scale, origin_degree)[0] + "1"
    )  # octave=1 just to get the pitch of the scales' tonic
    orig_tonic_pitch = pm.note_name_to_number(orig_tonic_note_name_str)

    # get the target tonic pitch
    target_tonic_note_name_str = (
        _get_chord_note_names(target_tonality, target_scale, target_degree)[0] + "1"
    )
    target_tonic_pitch = pm.note_name_to_number(target_tonic_note_name_str)

    diff_semitones = abs(origin_pitch - orig_tonic_pitch)
    target_pitch = diff_semitones + target_tonic_pitch

    # Check if target pitch corresponds to a note in the scale, if not,
    # convert the note to the closest note in the scale
    target_name = pm.note_number_to_name(target_pitch)
    target_name, target_octave = prettymidi_note_to_musicaiz(target_name)

    # Get the notes in the scale
    all_degs = ["I", "II", "III", "IV", "V", "VI", "VII"]
    notes_target_tonality = []
    for d in all_degs:
        notes_target_tonality.append(
            Tonality.get_chord_from_degree(target_tonality, d, target_scale)[0]
        )
    # notes_target_tonality = harmony.Tonality[target_tonality].notes
    notes_target_tonality = [note.name for note in notes_target_tonality]

    # get index in chromatic scale of the notes in the scale and the target note
    index_target_note = NoteClassBase[target_name].chromatic_scale_index
    indexes_scale_notes = [
        NoteClassBase[note].chromatic_scale_index
        for note in notes_target_tonality
    ]

    if index_target_note in indexes_scale_notes:
        return target_pitch
    else:
        # check the closest index of the note in the scale notes
        # Correct the note by assuming the correct one is the min closest note to the target note
        index_target = min(
            range(len(indexes_scale_notes)),
            key=lambda i: abs(indexes_scale_notes[i] - index_target_note),
        )
        # assign the target note name to the selected index
        name_target = NoteClassBase._get_note_from_chromatic_idx(
            indexes_scale_notes[index_target]
        )[0].name
        # name_target to pretty midi nomenclature
        name_target = NoteClassBase[name_target].value[0].contracted
        # convert the note name to the pitch with the octave
        target_pitch = pm.note_name_to_number(name_target + str(target_octave))
        return target_pitch


def _get_chord_note_names(tonality: str, scale: str, degree: str) -> List[str]:
    """Get the note name of a chord built from a degree of a scale."""
    chord_notes = Tonality.get_chord_notes_from_degree(
        tonality=tonality,
        degree=degree,
        scale=scale,
        chord_type="triad",
    )
    note_names_str = [note.value[0].contracted for note in chord_notes]
    return note_names_str
