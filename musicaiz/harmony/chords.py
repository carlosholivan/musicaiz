from __future__ import annotations
from enum import Enum
from typing import List, Optional, Union, Tuple, Dict
import re
import numpy as np
from pathlib import Path
import pretty_midi as pm


# Our modules
from .intervals import IntervalSemitones, Interval
from musicaiz.structure import (
    Note,
    NoteClassNames,
    NoteClassBase
)


class ChordComplexity(Enum):
    SIMPLE = 0
    COMPOUND = 1


class ChordInversion(Enum):
    FUNDAMENTAL = 0
    FIRST = 1
    SECOND = 2
    THIRD = 3
    FORTH = 4


class ChordQualities(Enum):
    """
    This Enum contains different nomenclatures for chord qualities.
    """
    # triads
    MINOR = ["m", "minor", "min", "-"]
    MAJOR = ["M", "major", "maj", "Δ"]
    AUGMENTED = ["A", "augmented", "aug", "+"]
    DIMINISHED = ["dis", "diminished", "dis", "°"]
    # 7ths
    MAJOR_SEVENTH = ["M7", "major seventh", "maj7", "Δ7", "Δ"]
    MINOR_SEVENTH = ["m7", "minor seventh", "-7"]
    DOMINANT_SEVENTH = ["7", "dominant seventh"]
    DIMINISHED_SEVENTH = ["dim7", "diminished seventh", "dim7", "°", "mb5", "-b5"]
    HALF_DIMINISHED_SEVENTH = ["m7b5", "half-diminished seventh", "-7b5", "ø"]
    MINOR_MAJOR_SEVENTH = ["mM7", "minor major seventh", "m maj7", "mΔ7", "-Δ7"]
    AUGMENTED_MAJOR_SEVENTH = ["maj7#5", "augmented major seventh", "+M7", "+Δ7"]
    AUGMENTED_SEVENTH = ["aug7", "augmented seventh", "+7"]
    DIMINISHED_MAJOR_SEVENTH = ["mM7b5", "diminished major seventh", "−Δ7b5"]
    DOMINANT_SEVENTH_FLAT_FIVE = ["7b5", "dominant seventh flat five"]
    MAJOR_SEVENTH_FLAT_FIVE = ["M7b5", "major seventh flat five"]
    # 7ths - 9ths
    # TODO: Poner propiedad para incluir la 7a o no
    # TODO: sus


    @classmethod
    def all_chord_qualities(cls) -> List[str]:
        chord_qualities = []
        for interval in cls.__members__.values():
            for name in interval.value:
                chord_qualities.append(name)
        return chord_qualities

    @classmethod
    def check_quality_exists(cls, name: str) -> bool:
        all_qualities = cls.all_chord_qualities()
        return name in all_qualities


class ChordType(Enum):
    """
    Number of notes (without duplicates) per chord type which also
    corresponds to the maximum number of inversions plus the
    root position of a chord.
    Ex.: a triad has 2 inversion plus the root position.
    """
    TRIAD = 3
    SEVENTH = 4
    NINTH = 5
    ELEVENTH = 6
    THIRTEENTH = 7

    @classmethod
    def get_type_from_value(cls, type_index: int) -> ChordType:
        """Retrieves the ChordType object given its type"""
        chord_obj = None
        for chord in cls.__members__.values():
            if chord.value == type_index:
                chord_obj = chord
        if chord_obj is None:
            raise ValueError(f"Chord {type} does not exist.")
        return chord_obj


# TODO: Add 9ths, 11yhs and 13ths (if proceed)
# TODO: Add property inversions to sort list of intervals?
class AllChords(Enum):
    """From root.
    Value 0: Nomenclature (quality).
    Value 1: List of intervals from root note.
    Value 2: Chord type (triad...)."""
    MAJOR_TRIAD = (
        ChordQualities.MAJOR,
        [
            IntervalSemitones.THIRD_MAJOR,
            IntervalSemitones.FIFTH_PERFECT,
        ],
        ChordType.TRIAD,
    )
    MINOR_TRIAD = (
        ChordQualities.MINOR,
        [
            IntervalSemitones.THIRD_MINOR,
            IntervalSemitones.FIFTH_PERFECT,
        ],
        ChordType.TRIAD,
    )
    AUGMENTED_TRIAD = (
        ChordQualities.AUGMENTED,
        [
            IntervalSemitones.THIRD_MAJOR,
            IntervalSemitones.FIFTH_AUGMENTED,
        ],
        ChordType.TRIAD,
    )
    DIMINISHED_TRIAD = (
        ChordQualities.DIMINISHED,
        [
            IntervalSemitones.THIRD_MINOR,
            IntervalSemitones.FIFTH_DIMINISHED,
        ],
        ChordType.TRIAD,
    )
    MAJOR_SEVENTH = (
        ChordQualities.MAJOR_SEVENTH,
        [
            IntervalSemitones.THIRD_MAJOR,
            IntervalSemitones.FIFTH_PERFECT,
            IntervalSemitones.SEVENTH_MAJOR
        ],
        ChordType.SEVENTH,
    )
    MINOR_SEVENTH = (
        ChordQualities.MINOR_SEVENTH,
        [
            IntervalSemitones.THIRD_MINOR,
            IntervalSemitones.FIFTH_PERFECT,
            IntervalSemitones.SEVENTH_MINOR
        ],
        ChordType.SEVENTH,
    )
    DOMINANT_SEVENTH = (
        ChordQualities.DOMINANT_SEVENTH,
        [
            IntervalSemitones.THIRD_MAJOR,
            IntervalSemitones.FIFTH_PERFECT,
            IntervalSemitones.SEVENTH_MINOR
        ],
        ChordType.SEVENTH,
    )
    DIMINISHED_SEVENTH = (
        ChordQualities.DIMINISHED_SEVENTH,
        [
            IntervalSemitones.THIRD_MINOR,
            IntervalSemitones.FIFTH_DIMINISHED,
            IntervalSemitones.SEVENTH_DIMINISHED
        ],
        ChordType.SEVENTH,
    )
    HALF_DIMINISHED_SEVENTH = (
        ChordQualities.HALF_DIMINISHED_SEVENTH,
        [
            IntervalSemitones.THIRD_MINOR,
            IntervalSemitones.FIFTH_DIMINISHED,
            IntervalSemitones.SEVENTH_MINOR
        ],
        ChordType.SEVENTH,
    )
    MINOR_MAJOR_SEVENTH = (
        ChordQualities.MINOR_MAJOR_SEVENTH,
        [
            IntervalSemitones.THIRD_MINOR,
            IntervalSemitones.FIFTH_PERFECT,
            IntervalSemitones.SEVENTH_MAJOR
        ],
        ChordType.SEVENTH,
    )
    AUGMENTED_MAJOR_SEVENTH = (
        ChordQualities.AUGMENTED_MAJOR_SEVENTH,
        [
            IntervalSemitones.THIRD_MAJOR,
            IntervalSemitones.FIFTH_AUGMENTED,
            IntervalSemitones.SEVENTH_MAJOR
        ],
        ChordType.SEVENTH,
    )
    AUGMENTED_SEVENTH = (
        ChordQualities.AUGMENTED_SEVENTH,
        [
            IntervalSemitones.THIRD_MAJOR,
            IntervalSemitones.FIFTH_AUGMENTED,
            IntervalSemitones.SEVENTH_MINOR
        ],
        ChordType.SEVENTH,
    )
    DIMINISHED_MAJOR_SEVENTH = (
        ChordQualities.DIMINISHED_MAJOR_SEVENTH,
        [
            IntervalSemitones.THIRD_MINOR,
            IntervalSemitones.FIFTH_DIMINISHED,
            IntervalSemitones.SEVENTH_MAJOR
        ],
        ChordType.SEVENTH,
    )
    DOMINANT_SEVENTH_FLAT_FIVE = (
        ChordQualities.DOMINANT_SEVENTH_FLAT_FIVE,
        [
            IntervalSemitones.THIRD_MAJOR,
            IntervalSemitones.FIFTH_DIMINISHED,
            IntervalSemitones.SEVENTH_MINOR
        ],
        ChordType.SEVENTH,
    )
    MAJOR_SEVENTH_FLAT_FIVE = (
        ChordQualities.MAJOR_SEVENTH_FLAT_FIVE,
        [
            IntervalSemitones.THIRD_MAJOR,
            IntervalSemitones.FIFTH_DIMINISHED,
            IntervalSemitones.SEVENTH_MAJOR
        ],
        ChordType.SEVENTH,
    )

    def __repr__(self):
        return "<%s.%s>" % (self.__class__.__name__, self.name)

    @property
    def chord_type(self):
        return self.value[2]

    @property
    def intervals(self):
        return self.value[1]

    @classmethod
    def get_all_qualities(cls) -> List[str]:
        all_notes = []
        for note in cls.__members__.values():
            for n in note.value:
                all_notes.append(n)
        return all_notes

    # TODO: Finish this
    @classmethod
    def get_chord_from_name(cls, chord_name: str) -> AllChords:
        """Retrieves the Chord object given its name"""
        chord_obj = None
        _, quality = Chord.split_chord_name(chord_name)
        for chord in cls.__members__.values():
            if quality in chord.value[0].value:
                chord_obj = chord
        if chord_obj is None:
            raise ValueError(f"Chord {chord_name} does not exist.")
        return chord_obj


class Chord:

    def __init__(self, chord: Optional[str] = None):
        """
        The Chord object can be initialized with a valid chord name
        or not. The logic behind this is having methods for generating and analyzing data.

        Initialization:
            - By initializing this class with a chord name, the use of this class' methods
                will be to generate data with the chord information.
            - By initializing this class with no input arguments, the use of this class' methods
                is more related to analize data (predict chords...).

        Raises
        ------
        ValueError
            if input chord name is invalid.

        Examples
        --------
        """

        self.chord_name = None
        self.root_note = None
        self.quality = None
        self.chord = None
        self.type = None
        if chord is not None:
            self.root_note, self.quality = self.split_chord_name(chord)
            self.chord = AllChords.get_chord_from_name(chord)
            self.quality_name = self.chord.value[0].value[1]
            self.type = self.chord.value[2]

    # TODO
    def get_notes(self, inversion: int = 0) -> List[str]:
        """Given the chord name and the root note, return all the notes in the chord.
        Arguments
        ---------
        inversion: int
            The chord inversion that will determined the notes order in the output list.
        """
        intervals = self.chord.value[1]
        root_note_obj = NoteClassBase.get_note_with_name(self.root_note)
        root_note_index = root_note_obj.value[2]
        notes = [self.root_note]
        for i in intervals:
            note_index = root_note_index + i.semitones
            note_index = note_index % 12
            quality = i.value[0]
            possible_notes = NoteClassBase._get_note_from_chromatic_idx(note_index)
            if len(possible_notes) == 1:
                #note_natural = Interval._initialize_note(note)
                append_note = possible_notes[0].value[0].contracted
            # TODO: select note A# or Bb depending on the chord
            else:
                append_note = possible_notes[0].value[0].contracted
            notes.append(append_note)
        # TODO: optimize this and test
        self._check_inversion_with_quality(inversion)
        for _ in range(inversion):
            notes.append(notes.pop(0))
        return notes

    def _check_inversion_with_quality(self, inversion: int):
        """Checks if the chord quality can be inverted to the input inversion.
        - Only higher than 7th chords do have a 3rd inversion.
        - Only higher than 9th chords do have a 4th inversion
        - Only higher than 11th chords do have a 5th inversion
        - Only higher than 13th chords do have a 6th inversion"""
        if self.type.value - 1 < inversion:
            raise ValueError(f"Chord quality {self.quality_name} does not have a {inversion} inversion.")

    @classmethod
    def get_all_chords(cls) -> Dict[str, Tuple[NoteClassBase, AllChords]]:
        """
        Constructs a dictionaray with the chord name (key) and its constructor
        (value). The constructor is a tuple with the chord's tonic and the quality.

        Output Ex.:
            {
                'C-MAJOR_TRIAD': (<NoteClassBase.C>, <AllChords.MAJOR_TRIAD>),
                ...
            }

        Returns:
            Dict[str, Tuple[NoteClassBase, AllChords]]
        """
        notes = NoteClassBase.get_notes_chromatic_scale(alteration = "SHARP")
        all_chords = {}
        exclude = [] #["SEVENTH", "AUGMENTED", "DIMINISHED"]
        for note in notes:
            tonic = note  # to avoid enharmonic chors such as C maj with Bb maj
            for chord in AllChords.__members__.values():
                if not any(e in chord.name for e in exclude):
                    all_chords.update({tonic.name + "-" + chord.name: (tonic, chord)})
        return all_chords

    @classmethod
    def get_notes_from_chord(
        cls,
        tonic: NoteClassBase,
        quality: AllChords,
    ) -> List[NoteClassBase]:
        """
        Generates the list of note objects that correspond to a chord.

        Parameters
        ----------

        tonic: NoteClassBase

        quality: AllChords


        Returns
        -------

        List[NoteClassBase]
        """
        notes = [tonic]
        for interval in quality.value[1]:
            interval_inst = Interval(interval)
            # Initialize note with name and octave
            degree_root = tonic.value[0].contracted + "1"
            note_obj = Interval._initialize_note(degree_root)
            note_dest_obj = interval_inst.transpose_note(note_obj)
            notes.append(note_dest_obj.note)
        return notes

    @classmethod
    def get_pitches_from_chord(
        cls,
        tonic: NoteClassBase,
        quality: AllChords,
        octave: int,
    ) -> List[int]:
        """
        Generates the list of pitches that correspond to a chord and an octave.
        The octaves go from -1 (pitch = 0) to 11 (pitch = 127)

        Parameters
        ----------

        tonic: NoteClassBase

        quality: AllChords

        octave: int

        Returns
        -------

        List[int]
        """
        notes = cls.get_notes_from_chord(tonic, quality)
        return [pm.note_name_to_number(note.value[0].contracted + str(octave)) for note in notes]
    

    @classmethod
    def chords_to_onehot(cls) -> Dict[str, List[int]]:
        """
        Converts the chords to a one hot representation.

        Returns
        -------

        chord_pitches_dict: Dict[str, List[int]]
        """
        start_octave = -1
        all_chords = cls.get_all_chords()
        chord_pitches_dict = {}
        for chord in all_chords.keys():
            chord_pitches = cls.get_pitches_from_chord(all_chords[chord][0], all_chords[chord][1], start_octave)
            chord_freqs = [1 if i in chord_pitches else 0 for i in range(0, 12)] #[pm.note_number_to_hz(pitch) for pitch in chord_pitches]
            chord_pitches_dict.update({chord: chord_freqs})
        return chord_pitches_dict

    @staticmethod
    def split_chord_name(chord_name: str) -> Tuple[str, str]:
        """Splits a chord name to its root note and the chord quality."""
        chord_data = [i for i in re.split(r'([A-G#b]+)', chord_name) if i]
        if len(chord_data) == 0:
            raise ValueError("Invalid chord {chord_name}.")
        root_note = chord_data[0]
        quality = ''.join(i for i in chord_data[1:])
        if not NoteClassNames.check_note_name_exists(root_note):
            raise ValueError(f"Root note {root_note} does not exist.")
        if not ChordQualities.check_quality_exists(quality):
            raise ValueError(f"Chord quality {quality} does not exist.")
        return root_note, quality

    def __repr__(self):
        return "Chord(root_note={}, quality='{}', type={}, name='{}')".format(
            self.root_note,
            self.quality,
            self.type.name.lower(),
            self.quality_name,
        )
