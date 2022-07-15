from __future__ import annotations
import re
from typing import Union, Optional, Tuple, List
from enum import Enum
import pretty_midi as pm


# Our modules
from musicaiz.structure import (
    NoteClassBase,
    NoteValue,
    NoteTiming,
    Note,
)


# TODO: Allow compound intervals (reduce them to simple intervals)
class IntervalClass(Enum):
    UNISON = "1"
    SECOND = "2"
    THIRD = "3"
    FOURTH = "4"
    FIFTH = "5"
    SIXTH = "6"
    SEVENTH = "7"
    OCTAVE = "8"

    @property
    def quality(self):
        return self.value

    @staticmethod
    def get_all_interval_classes() -> List[str]:
        return [interval.value for interval in IntervalClass]


# TODO: Add here DOUBLY_AUGMENTED and DOUBLY_DIMINISHED
class IntervalQuality(Enum):
    MINOR = ["m", "minor", "min", "-"]
    MAJOR = ["M", "major", "maj", "Δ"]
    AUGMENTED = ["A", "augmented", "aug", "+"]
    DIMINISHED = ["dim", "diminished", "dim", "°"]
    PERFECT = ["P", "perfect", "", ""]

    @property
    def large(self) -> str:
        return self.value[1]

    @property
    def medium(self) -> str:
        return self.value[2]

    @property
    def contracted(self) -> str:
        return self.value[0]

    @property
    def symbol(self) -> str:
        return self.value[3]

    @staticmethod
    def get_all_interval_qualities() -> List[str]:
        qualities = []
        for quality in IntervalClass:
            for q in quality.value:
                if q not in qualities:
                    qualities.append(q)
        return qualities


class IntervalSemitones(Enum):
    UNISON_PERFECT = (IntervalClass.UNISON, IntervalQuality.PERFECT, 0)
    UNISON_AUGMENTED = (IntervalClass.UNISON, IntervalQuality.AUGMENTED, 1)
    UNISON_DOUBLY_AUGMENTED = (IntervalClass.UNISON, IntervalQuality.AUGMENTED, 2)
    SECOND_DIMINISHED = (IntervalClass.SECOND, IntervalQuality.DIMINISHED, 0)
    SECOND_MINOR = (IntervalClass.SECOND, IntervalQuality.MINOR, 1)
    SECOND_MAJOR = (IntervalClass.SECOND, IntervalQuality.MAJOR, 2)
    SECOND_AUGMENTED = (IntervalClass.SECOND, IntervalQuality.AUGMENTED, 3)
    SECOND_DOUBLY_AUGMENTED = (IntervalClass.SECOND, IntervalQuality.AUGMENTED, 4)
    THIRD_DOUBLY_DIMINISHED = (IntervalClass.THIRD, IntervalQuality.DIMINISHED, 1)
    THIRD_DIMINISHED = (IntervalClass.THIRD, IntervalQuality.DIMINISHED, 2)
    THIRD_MINOR = (IntervalClass.THIRD, IntervalQuality.MINOR, 3)
    THIRD_MAJOR = (IntervalClass.THIRD, IntervalQuality.MAJOR, 4)
    THIRD_AUGMENTED = (IntervalClass.THIRD, IntervalQuality.AUGMENTED, 5)
    THIRD_DOUBLY_AUGMENTED = (IntervalClass.THIRD, IntervalQuality.AUGMENTED, 6)
    FOURTH_DOUBLY_DIMINISHED = (IntervalClass.FOURTH, IntervalQuality.DIMINISHED, 3)
    FOURTH_DIMINISHED = (IntervalClass.FOURTH, IntervalQuality.DIMINISHED, 4)
    FOURTH_PERFECT = (IntervalClass.FOURTH, IntervalQuality.PERFECT, 5)
    FOURTH_AUGMENTED = (IntervalClass.FOURTH, IntervalQuality.AUGMENTED, 6)
    FOURTH_DOUBLY_AUGMENTED = (IntervalClass.FOURTH, IntervalQuality.AUGMENTED, 7)
    FIFTH_PERFECT = (IntervalClass.FIFTH, IntervalQuality.PERFECT, 7)
    FIFTH_DOUBLY_DIMINISHED = (IntervalClass.FIFTH, IntervalQuality.DIMINISHED, 5)
    FIFTH_DIMINISHED = (IntervalClass.FIFTH, IntervalQuality.DIMINISHED, 6)
    FIFTH_AUGMENTED = (IntervalClass.FIFTH, IntervalQuality.AUGMENTED, 8)
    FIFTH_DOUBLY_AUGMENTED = (IntervalClass.FIFTH, IntervalQuality.AUGMENTED, 9)
    SIXTH_DIMINISHED = (IntervalClass.SIXTH, IntervalQuality.DIMINISHED, 6)
    SIXTH_DOUBLY_DIMINISHED = (IntervalClass.SIXTH, IntervalQuality.DIMINISHED, 7)
    SIXTH_MINOR = (IntervalClass.SIXTH, IntervalQuality.MINOR, 8)
    SIXTH_MAJOR = (IntervalClass.SIXTH, IntervalQuality.MAJOR, 9)
    SIXTH_AUGMENTED = (IntervalClass.SIXTH, IntervalQuality.AUGMENTED, 10)
    SIXTH_DOUBLY_AUGMENTED = (IntervalClass.SIXTH, IntervalQuality.AUGMENTED, 11)
    SEVENTH_DOUBLY_DIMINISHED = (IntervalClass.SEVENTH, IntervalQuality.DIMINISHED, 8)
    SEVENTH_DIMINISHED = (IntervalClass.SEVENTH, IntervalQuality.DIMINISHED, 9)
    SEVENTH_MINOR = (IntervalClass.SEVENTH, IntervalQuality.MINOR, 10)
    SEVENTH_MAJOR = (IntervalClass.SEVENTH, IntervalQuality.MAJOR, 11)
    SEVENTH_AUGMENTED = (IntervalClass.SEVENTH, IntervalQuality.AUGMENTED, 12)
    SEVENTH_DOUBLY_AUGMENTED = (IntervalClass.SEVENTH, IntervalQuality.AUGMENTED, 13)
    OCTAVE_DIMINISHED = (IntervalClass.OCTAVE, IntervalQuality.DIMINISHED, 11)
    OCTAVE_PERFECT = (IntervalClass.OCTAVE, IntervalQuality.PERFECT, 12)

    def __repr__(self):
        return "<%s.%s>" % (self.__class__.__name__, self.name)

    @property
    def semitones(self) -> int:
        return self.value[2]

    @property
    def names(self) -> int:
        return [self.value[0].quality + i for i in self.value[1].value]

    @property
    def large(self) -> str:
        return self.value[0].quality + self.value[1].large

    @property
    def medium(self) -> str:
        return self.value[0].quality + self.value[1].medium

    @property
    def contracted(self) -> str:
        return self.value[0].quality + self.value[1].contracted

    @property
    def symbol(self) -> str:
        return self.value[0].quality + self.value[1].symbol

    @classmethod
    def all_interval_names(cls) -> str:
        interval_names = []
        for interval in cls.__members__.values():
            interval_names.append(interval.large)
            interval_names.append(interval.medium)
            interval_names.append(interval.contracted)
            interval_names.append(interval.symbol)
        return interval_names

    @classmethod
    def check_interval_exists(cls, name: str) -> bool:
        interval_names = cls.all_interval_names()
        return name in interval_names

    @classmethod
    def get_interval_from_semitones(cls, semitones: int) -> List[IntervalSemitones]:
        possible_intervals = []
        for interval in cls.__members__.values():
            if semitones == interval.value[2]:
                possible_intervals.append(interval)
        if len(possible_intervals) == 0:
            raise ValueError("Not interval found for the input semitones.")
        return possible_intervals

    @classmethod
    def get_qualities_from_semitones(cls, semitones: int) -> IntervalQuality:
        """Returns a list of all possible interval qualities for the
        input semitones."""
        return cls._get_qualities_classes_from_semitones(semitones)[0]

    @classmethod
    def get_classes_from_semitones(cls, semitones: int) -> IntervalQuality:
        """Returns a list of all possible interval classes for the
        input semitones."""
        return cls._get_qualities_classes_from_semitones(semitones)[1]

    @classmethod
    def get_class_from_quality_semitones(cls, quality: str, semitones: int) -> IntervalQuality:
        """Returns the only interval class that blongs to the input quality and semitones.
        Raises
        ------
            ValueError
                if the combination of semitones and quality does not match any interval"""
        intervals = cls.get_interval_from_semitones(semitones)
        interval_class = None
        for interval in intervals:
            if quality in interval.value[1].value and semitones == interval.value[2]:
                interval_class = interval.value[0]
        if interval_class is None:
            raise ValueError("Interval class not fuond for the input quality and semitones.")
        return interval_class

    @classmethod
    def get_quality_from_class_semitones(cls, interval_class: str, semitones: int) -> IntervalQuality:
        """Returns the only interval class that blongs to the input quality and semitones.
        Raises
        ------
            ValueError
                if the combination of semitones and quality does not match any interval"""
        intervals = cls.get_interval_from_semitones(semitones)
        quality = None
        for interval in intervals:
            if interval_class == interval.value[0].value and semitones == interval.value[2]:
                quality = interval.value[1]
        if quality is None:
            raise ValueError("Interval class not found for the input quality and semitones.")
        return quality

    @classmethod
    def get_interval_from_name(cls, interval: str) -> IntervalSemitones:
        interval_obj = None
        interval_data = [i for i in re.split(r'([A-Za-z]+)', interval) if i]
        interval_class = interval_data[0]
        quality = interval_data[1]
        for interval in cls.__members__.values():
            if interval_class == interval.value[0].value and quality in interval.value[1].value:
                interval_obj = interval
        if interval_obj is None:
            raise ValueError(f"Interval {interval} does not exist.")
        return interval_obj

    @classmethod
    def _get_qualities_classes_from_semitones(cls, semitones: int) -> IntervalQuality:
        """Returns a list of all possible interval classes and qualities for the
        input semitones.
        Raises
        ------
            ValueError
                if semitones cannot be found in the intervals."""
        possible_intervals = cls.get_interval_from_semitones(semitones)
        classes = []
        qualities = []
        for interval in possible_intervals:
            classes.append(interval.value[0])
            qualities.append(interval.value[1])
        if len(classes) == 0 or len(qualities) == 0:
            raise ValueError("Not valid semitones value.")
        return qualities, classes


# TODO: I don't know yet where I'll use this, but this is useful
class IntervalComplexity(Enum):
    SIMPLE = 0  # interval class is more than an onctave
    COMPOUND = 1  # interval class is less than an onctave


class Interval:

    __slots__ = [
        "interval",
        "interval_class",
        "quality",
        "semitones",
        "form"
    ]

    def __init__(self, interval: Optional[Union[IntervalSemitones, str]] = None):

        """
        The Interval object can be initialized with a valid interval name
        or not. The logic behind this is having methods for generating and analyzing data.

        Initialization:
            - By initializing this class with an interval name, the use of this class' methods
                will be to generate data with the interval information.
            - By initializing this class with no input arguments, the use of this class' methods
                is more related to analize data.

        Raises
        ------
        ValueError
            if input interval name is invalid.

        Examples
        --------
        - Get interval between 2 notes

        >>> interval = Interval()
        >>> note1 = "C0"
        >>> note2 = "G0"
        >>> interval_name = interval.get_interval(note1, note2)
        """

        self.interval = None
        self.interval_class = None
        self.quality = None
        self.semitones = None
        self.form = None

        if interval is not None:
            all_intervals = IntervalSemitones.all_interval_names()
            if isinstance(interval, str):
                if interval not in all_intervals:
                    raise ValueError(f"Input interval {interval} is not a valid interval.")
                interval_obj = IntervalSemitones.get_interval_from_name(interval)
            else:
                if interval not in IntervalSemitones.__members__.values():
                    raise ValueError(f"Input interval {interval} is not a valid interval.")
                interval_obj = interval

            self.interval = interval_obj
            self.interval_class = interval_obj.value[0]
            self.quality = interval_obj.value[1]
            self.semitones = interval_obj.value[2]
            if int(self.interval_class.value) > 8:
                self.form = "compound"
            else:
                self.form = "simple"

    def transpose_note(
        self,
        note: Union[str, int, Note, NoteValue, NoteTiming],
    ) -> Union[Note, NoteValue, NoteTiming]:
        """Transposes the input note a given interval."""
        note = self._initialize_note(note)
        new_pitch = note.pitch + self.semitones
        return NoteValue(new_pitch)

    @staticmethod
    def _initialize_note(
        note: Union[str, int, Note, NoteValue, NoteTiming]
    ) -> Union[Note, NoteValue, NoteTiming]:
        if isinstance(note, (int, str)):
            return NoteValue(note)
        else:
            return note

    @classmethod
    def get_possible_intervals(
        cls,
        note1: Union[str, int, Note],
        note2: Union[str, int, Note]
    ) -> List[Tuple[str, IntervalSemitones]]:
        """
        Calculates the interval for a given pair of notes.
        This takes into account also the enharmonic notes because if we try to predict
        intervals from MIDIs, we do know the pitch of the notes but not the note names.
        Ex.: 24 is C but also B#. Depending on the scale which we don't know in MIDI files,
        the pitch will  correspond to C or B#.
        The interval will be calculated in its simplest form. That means that if the notes
        form a compound interval (distances higher than an 8ve), the notes
        will be by rooted to the same octave.
        Important: We measure the intervals in ascendent order.

        Returns:
            [type]: [description]
        """
        intervals = []
        # Initialize note objects if inputs note is int or str
        note1 = cls._initialize_note(note1)
        note2 = cls._initialize_note(note2)
        note1_enh = cls._initialize_note(note1.enharmonic.value[0].value[0] + note1.octave)
        note2_enh = cls._initialize_note(note2.enharmonic.value[0].value[0] + note2.octave)

        # Build the notes combinations.
        # When enharmonic note is the same note as the base note we don't build a combination.
        # Ex.: D has no enharmonics, so if note1 is D, then note1_enh will be also D.
        # TODO: Helper method for this
        combinations = []
        combinations.append((note1, note2))
        if note1.note_name != note1_enh.note_name:
            combinations.append((note1_enh, note2))
        if note2.note_name != note2_enh.note_name:
            combinations.append((note1, note2_enh))
        if note1.note_name != note1_enh.note_name and note2.note_name != note2_enh.note_name:
            combinations.append((note1_enh, note2_enh))

        # TODO: Refactor this to another method `get_interval` that takes 2 note names C#...
        for i in range(len(combinations)):
            note_names_str = combinations[i][0].note_name + "-" + combinations[i][1].note_name
            # Get notes indexes in the chromatic scale to obtain their distance
            note_1_index = combinations[i][0].note.value[2]
            note_2_index = combinations[i][1].note.value[2]

            # we'll know the exact interval class by measuring the distance
            # between the notes in the natural scale
            # We convert the notes in their natural scale form (`C#` -> `C`)
            natural_note1, _ = NoteValue.split_note_name(combinations[i][0].note_name)
            natural_note2, _ = NoteValue.split_note_name(combinations[i][1].note_name)

            # Get the distance (class) of the interval with the semitones diff in hte natural scale
            note1_natural_index = NoteClassBase[natural_note1].value[1]
            note2_natural_index = NoteClassBase[natural_note2].value[1]

            # if the notes are exactly the same (unison), they're in perfect octave
            if note_2_index == note_1_index and natural_note1 == natural_note2:
                if combinations[i][0].pitch == combinations[i][1].pitch:
                    intervals.append((note_names_str, IntervalSemitones.UNISON_PERFECT))
                else:
                    intervals.append((note_names_str, IntervalSemitones.OCTAVE_PERFECT))
                continue
            if abs(note_2_index - note_1_index) == 1:
                if note1_natural_index - note2_natural_index == 0:
                    intervals.append((note_names_str, IntervalSemitones.UNISON_AUGMENTED))
                else:
                    intervals.append((note_names_str, IntervalSemitones.SECOND_MINOR))
                continue
            # If our interval has a C in the middle of its path, we cont +1 in the quality
            # D-G is a 4th but G-D is a 5th bc we have C in the middle from G-D but not from D-G
            elif note1_natural_index > note2_natural_index:
                interval_class = str(1 + note2_natural_index + 7 - note1_natural_index)
                semitones_scale_diff = (note_2_index + 12) - note_1_index
            else:
                if combinations[i][0].note == NoteClassBase.B_SHARP or combinations[i][1].note == NoteClassBase.B_SHARP:
                    # Get the possible interval classes with the distances in semitones
                    # In the case note2 is lower, as we have C in the middle we traspose its idx 1 8ve
                    interval_class = str(1 + abs(note1_natural_index - note2_natural_index))
                    semitones_scale_diff = (note_2_index + 12) - note_1_index
                else:
                    semitones_scale_diff = abs(note_2_index - note_1_index)
                    interval_class = str(1 + abs(note1_natural_index - note2_natural_index))
            # We finally match the interval_class and the semitones diff to get the interval
            for j in IntervalSemitones:
                if interval_class == j.value[0].value and semitones_scale_diff == j.value[2]:
                    interval = j
                    break
            intervals.append((note_names_str, interval))
        return intervals

    def __repr__(self):
        return "Interval(interval={}, interval_class={}, quality='{}', semitones={}, form={})".format(
            self.interval,
            self.interval_class,
            self.quality,
            self.semitones,
            self.form,
        )
