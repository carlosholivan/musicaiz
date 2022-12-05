from __future__ import annotations
from enum import Enum
from typing import List, Tuple, Optional, Union
import copy


# Our modules
from musicaiz.structure import (
    NoteClassBase,
    AccidentalsNames
)
from musicaiz.harmony import (
    AllChords,
    Interval,
)


"""
https://www.dolmetsch.com/musictheory17.htm
"""


class AccidentalsNum(Enum):
    ONE_SHARP = 1
    TWO_SHARPS = 2


class DegreesQualities(Enum):
    """
    This class defines the qualities of the degree's chords
    from triads to 11th chords.

    Properties:
        contracted: contracted notation.
        large: full description.
    """
    # triads
    MINOR = ["", "minor"]
    MAJOR = ["", "major"]
    AUGMENTED = ["+", "augmented"]
    DIMINISHED = ["o", "diminished"]
    # 7ths
    MAJOR_SEVENTH = ["M7", "major seventh"]
    MINOR_SEVENTH = ["7", "minor seventh"]
    # TODO: finnish this
    DOMINANT_SEVENTH = ["7", "dominant seventh"]
    DIMINISHED_SEVENTH = ["ø7", "diminished seventh"]
    HALF_DIMINISHED_SEVENTH = ["ø7", "half-diminished seventh"]

    MINOR_MAJOR_SEVENTH = ["mM7", "minor major seventh", "m maj7", "mΔ7", "-Δ7"]
    AUGMENTED_MAJOR_SEVENTH = ["maj7#5", "augmented major seventh", "+M7", "+Δ7"]
    AUGMENTED_SEVENTH = ["aug7", "augmented seventh", "+7"]
    DIMINISHED_MAJOR_SEVENTH = ["mM7b5", "diminished major seventh", "−Δ7b5"]
    DOMINANT_SEVENTH_FLAT_FIVE = ["7b5", "dominant seventh flat five"]
    MAJOR_SEVENTH_FLAT_FIVE = ["M7b5", "major seventh flat five"]

    @property
    def contracted(self) -> str:
        return self.value[0]

    @property
    def large(self) -> str:
        return self.value[1]


class DegreesRoman(Enum):
    FIRST = "I"
    SECOND = "II"
    THIRD = "III"
    FOURTH = "IV"
    FIFTH = "V"
    SIXTH = "VI"
    SEVENTH = "VII"

    @property
    def major(self) -> str:
        return self.value

    @property
    def minor(self) -> str:
        return self.value.lower()

    @property
    def diminished(self) -> str:
        return self.value.lower() + "ø"

    @property
    def half_diminished():
        pass

    @property
    def index(self) -> int:
        for i, deg in enumerate(DegreesRoman.__members__.values()):
            if self.value == deg.value:
                return i

    @staticmethod
    def get_name_with_degree(degree: str) -> DegreesRoman:
        for i in DegreesRoman.__members__.values():
            if i.value == degree:
                return i

    @classmethod
    def get_degree_from_index(cls, degree_index: int) -> DegreesRoman:
        for i, deg in enumerate(DegreesRoman.__members__.values()):
            if i == degree_index:
                return deg


class Degrees:
    """Parent class for degrees naming conventions."""

    @property
    def contracted_name(self):
        return self.value[1] + self.value[2].contracted

    @property
    def large_name(self):
        return self.value[1] + self.value[2].large

    @property
    def chord(self):
        return self.value[0]

    def __repr__(self):
        return "<%s.%s>" % (self.__class__.__name__, self.name)


# ===================Triads========================
class MajorTriadDegrees(Degrees, Enum):
    FIRST = (
        AllChords.MAJOR_TRIAD,
        DegreesRoman.FIRST.major,
        DegreesQualities.MAJOR
    )
    SECOND = (
        AllChords.MINOR_TRIAD,
        DegreesRoman.SECOND.minor,
        DegreesQualities.MINOR
    )
    THIRD = (
        AllChords.MINOR_TRIAD,
        DegreesRoman.THIRD.minor,
        DegreesQualities.MINOR
    )
    FOURTH = (
        AllChords.MAJOR_TRIAD,
        DegreesRoman.FOURTH.major,
        DegreesQualities.MAJOR
    )
    FIFTH = (
        AllChords.MAJOR_TRIAD,
        DegreesRoman.FIFTH.major,
        DegreesQualities.MAJOR
    )
    SIXTH = (
        AllChords.MINOR_TRIAD,
        DegreesRoman.SIXTH.minor,
        DegreesQualities.MINOR
    )
    SEVENTH = (
        AllChords.DIMINISHED_TRIAD,
        DegreesRoman.SEVENTH.diminished,
        DegreesQualities.DIMINISHED
    )


class MinorNaturalTriadDegrees(Degrees, Enum):
    FIRST = (
        AllChords.MINOR_TRIAD,
        DegreesRoman.FIRST.minor,
        DegreesQualities.MINOR
    )
    SECOND = (
        AllChords.DIMINISHED_TRIAD,
        DegreesRoman.SECOND.minor,
        DegreesQualities.DIMINISHED
    )
    THIRD = (
        AllChords.MAJOR_TRIAD,
        "b" + DegreesRoman.THIRD.major,
        DegreesQualities.MAJOR
    )
    FOURTH = (
        AllChords.MINOR_TRIAD,
        DegreesRoman.FOURTH.minor,
        DegreesQualities.MINOR
    )
    FIFTH = (
        AllChords.MINOR_TRIAD,
        DegreesRoman.FIFTH.minor,
        DegreesQualities.MINOR
    )
    SIXTH = (
        AllChords.MAJOR_TRIAD,
        "b" + DegreesRoman.SIXTH.major,
        DegreesQualities.MAJOR
    )
    SEVENTH = (
        AllChords.MAJOR_TRIAD,
        "b" + DegreesRoman.SEVENTH.major,
        DegreesQualities.MAJOR
    )


class MinorHarmonicTriadDegrees(Degrees, Enum):
    FIRST = (
        AllChords.MINOR_TRIAD,
        DegreesRoman.FIRST.minor,
        DegreesQualities.MINOR
    )
    SECOND = (
        AllChords.DIMINISHED_TRIAD,
        DegreesRoman.SECOND.minor,
        DegreesQualities.DIMINISHED
    )
    THIRD = (
        AllChords.AUGMENTED_TRIAD,
        DegreesRoman.THIRD.major,
        DegreesQualities.AUGMENTED
    )
    FOURTH = (
        AllChords.MINOR_TRIAD,
        DegreesRoman.FOURTH.minor,
        DegreesQualities.MINOR
    )
    FIFTH = (
        AllChords.MAJOR_TRIAD,
        DegreesRoman.FIFTH.major,
        DegreesQualities.MAJOR
    )
    SIXTH = (
        AllChords.MAJOR_TRIAD,
        DegreesRoman.SIXTH.major,
        DegreesQualities.MAJOR
    )
    SEVENTH = (
        AllChords.DIMINISHED_TRIAD,
        DegreesRoman.SEVENTH.minor,
        DegreesQualities.DIMINISHED
    )


class MinorMelodicTriadDegrees(Degrees, Enum):
    FIRST = (
        AllChords.MINOR_TRIAD,
        DegreesRoman.FIRST.minor,
        DegreesQualities.MINOR
    )
    SECOND = (
        AllChords.MINOR_TRIAD,
        DegreesRoman.SECOND.minor,
        DegreesQualities.MINOR
    )
    THIRD = (
        AllChords.AUGMENTED_TRIAD,
        DegreesRoman.THIRD.major,
        DegreesQualities.AUGMENTED
    )
    FOURTH = (
        AllChords.MAJOR_TRIAD,
        DegreesRoman.FOURTH.major,
        DegreesQualities.MAJOR
    )
    FIFTH = (
        AllChords.MAJOR_TRIAD,
        DegreesRoman.FIFTH.major,
        DegreesQualities.MAJOR
    )
    SIXTH = (
        AllChords.DIMINISHED_TRIAD,
        DegreesRoman.SIXTH.minor,
        DegreesQualities.DIMINISHED
    )
    SEVENTH = (
        AllChords.DIMINISHED_TRIAD,
        DegreesRoman.SEVENTH.minor,
        DegreesQualities.DIMINISHED
    )


class DorianTriadDegrees(Degrees, Enum):
    FIRST = (
        AllChords.MINOR_TRIAD,
        DegreesRoman.FIRST.minor,
        DegreesQualities.MINOR
    )
    SECOND = (
        AllChords.MINOR_TRIAD,
        DegreesRoman.SECOND.minor,
        DegreesQualities.MINOR
    )
    THIRD = (
        AllChords.MAJOR_TRIAD,
        "b" + DegreesRoman.THIRD.major,
        DegreesQualities.MAJOR
    )
    FOURTH = (
        AllChords.MAJOR_TRIAD,
        DegreesRoman.FOURTH.major,
        DegreesQualities.MAJOR
    )
    FIFTH = (
        AllChords.MINOR_TRIAD,
        DegreesRoman.FIFTH.minor,
        DegreesQualities.MINOR
    )
    SIXTH = (
        AllChords.DIMINISHED_TRIAD,
        DegreesRoman.SIXTH.minor,
        DegreesQualities.DIMINISHED
    )
    SEVENTH = (
        AllChords.MAJOR_TRIAD,
        "b" + DegreesRoman.SEVENTH.major,
        DegreesQualities.MAJOR
    )


class PhrygianTriadDegrees(Degrees, Enum):
    FIRST = (
        AllChords.MINOR_TRIAD,
        DegreesRoman.FIRST.minor,
        DegreesQualities.MINOR
    )
    SECOND = (
        AllChords.MAJOR_TRIAD,
        "b" + DegreesRoman.SECOND.major,
        DegreesQualities.MAJOR
    )
    THIRD = (
        AllChords.MAJOR_TRIAD,
        "b" + DegreesRoman.THIRD.major,
        DegreesQualities.MAJOR
    )
    FOURTH = (
        AllChords.MINOR_TRIAD,
        DegreesRoman.FOURTH.minor,
        DegreesQualities.MINOR
    )
    FIFTH = (
        AllChords.DIMINISHED_TRIAD,
        DegreesRoman.FIFTH.minor,
        DegreesQualities.DIMINISHED
    )
    SIXTH = (
        AllChords.MAJOR_TRIAD,
        "b" + DegreesRoman.SIXTH.major,
        DegreesQualities.MAJOR
    )
    SEVENTH = (
        AllChords.MINOR_TRIAD,
        "b" + DegreesRoman.SEVENTH.minor,
        DegreesQualities.MINOR
    )


class LydianTriadDegrees(Degrees, Enum):
    FIRST = (
        AllChords.MAJOR_TRIAD,
        DegreesRoman.FIRST.major,
        DegreesQualities.MAJOR
    )
    SECOND = (
        AllChords.MAJOR_TRIAD,
        DegreesRoman.SECOND.major,
        DegreesQualities.MAJOR
    )
    THIRD = (
        AllChords.MINOR_TRIAD,
        "b" + DegreesRoman.THIRD.minor,
        DegreesQualities.MINOR
    )
    FOURTH = (
        AllChords.DIMINISHED_TRIAD,
        "#" + DegreesRoman.FOURTH.minor,
        DegreesQualities.DIMINISHED
    )
    FIFTH = (
        AllChords.MAJOR_TRIAD,
        DegreesRoman.FIFTH.major,
        DegreesQualities.MAJOR
    )
    SIXTH = (
        AllChords.MINOR_TRIAD,
        "b" + DegreesRoman.SIXTH.minor,
        DegreesQualities.MINOR
    )
    SEVENTH = (
        AllChords.MINOR_TRIAD,
        "b" + DegreesRoman.SEVENTH.minor,
        DegreesQualities.MINOR
    )


class MixolydianTriadDegrees(Degrees, Enum):
    FIRST = (
        AllChords.MAJOR_TRIAD,
        DegreesRoman.FIRST.major,
        DegreesQualities.MAJOR
    )
    SECOND = (
        AllChords.MINOR_TRIAD,
        DegreesRoman.SECOND.minor,
        DegreesQualities.MINOR
    )
    THIRD = (
        AllChords.DIMINISHED_TRIAD,
        DegreesRoman.THIRD.minor,
        DegreesQualities.DIMINISHED
    )
    FOURTH = (
        AllChords.MAJOR_TRIAD,
        DegreesRoman.FOURTH.major,
        DegreesQualities.MAJOR
    )
    FIFTH = (
        AllChords.MINOR_TRIAD,
        DegreesRoman.FIFTH.minor,
        DegreesQualities.MINOR
    )
    SIXTH = (
        AllChords.MINOR_TRIAD,
        "b" + DegreesRoman.SIXTH.minor,
        DegreesQualities.MINOR
    )
    SEVENTH = (
        AllChords.MAJOR_TRIAD,
        "b" + DegreesRoman.SEVENTH.major,
        DegreesQualities.MAJOR
    )


class LocrianTriadDegrees(Degrees, Enum):
    FIRST = (
        AllChords.DIMINISHED_TRIAD,
        DegreesRoman.FIRST.minor,
        DegreesQualities.DIMINISHED
    )
    SECOND = (
        AllChords.MAJOR_TRIAD,
        "b" + DegreesRoman.SECOND.major,
        DegreesQualities.MAJOR
    )
    THIRD = (
        AllChords.MINOR_TRIAD,
        "b" + DegreesRoman.THIRD.minor,
        DegreesQualities.MINOR
    )
    FOURTH = (
        AllChords.MINOR_TRIAD,
        DegreesRoman.FOURTH.minor,
        DegreesQualities.MINOR
    )
    FIFTH = (
        AllChords.MAJOR_TRIAD,
        "b" + DegreesRoman.FIFTH.major,
        DegreesQualities.MAJOR
    )
    SIXTH = (
        AllChords.MAJOR_TRIAD,
        "b" + DegreesRoman.SIXTH.major,
        DegreesQualities.MAJOR
    )
    SEVENTH = (
        AllChords.MINOR_TRIAD,
        "b" + DegreesRoman.SEVENTH.minor,
        DegreesQualities.MINOR
    )


class TriadsModes(Enum):
    MAJOR = [el for el in MajorTriadDegrees]
    MINOR_NATURAL = [el for el in MinorNaturalTriadDegrees]
    MINOR_HARMONIC = [el for el in MinorHarmonicTriadDegrees]
    MINOR_MELODIC = [el for el in MinorMelodicTriadDegrees]
    # Greek modes
    IONIAN = MAJOR
    DORIAN = [el for el in DorianTriadDegrees]
    PHRYGIAN = [el for el in PhrygianTriadDegrees]
    LYDIAN = [el for el in LydianTriadDegrees]
    MIXOLYDIAN = [el for el in MixolydianTriadDegrees]
    AEOLIAN = MINOR_NATURAL
    LOCRIAN = [el for el in LocrianTriadDegrees]

    def degree(self, degree_index: int) -> str:
        return self.value[degree_index + 1][1]


# ===================7ths========================
class MajorSeventhDegrees(Degrees, Enum):
    FIRST = (
        AllChords.MAJOR_SEVENTH,
        DegreesRoman.FIRST.major,
        DegreesQualities.MAJOR_SEVENTH
    )
    SECOND = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.SECOND.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    THIRD = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.THIRD.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    FOURTH = (
        AllChords.MAJOR_SEVENTH,
        DegreesRoman.FOURTH.major,
        DegreesQualities.MAJOR_SEVENTH
    )
    FIFTH = (
        AllChords.DOMINANT_SEVENTH,
        DegreesRoman.FIFTH.major,
        DegreesQualities.DOMINANT_SEVENTH
    )
    SIXTH = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.SIXTH.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    SEVENTH = (
        AllChords.DIMINISHED_SEVENTH,
        DegreesRoman.SEVENTH.diminished,
        DegreesQualities.DIMINISHED_SEVENTH
    )


class MinorNaturalSeventhDegrees(Degrees, Enum):
    FIRST = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.FIRST.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    SECOND = (
        AllChords.HALF_DIMINISHED_SEVENTH,
        DegreesRoman.SECOND.minor,
        DegreesQualities.HALF_DIMINISHED_SEVENTH
    )
    THIRD = (
        AllChords.MAJOR_SEVENTH,
        "b" + DegreesRoman.THIRD.major,
        DegreesQualities.MAJOR_SEVENTH
    )
    FOURTH = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.FOURTH.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    FIFTH = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.FIFTH.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    SIXTH = (
        AllChords.MAJOR_SEVENTH,
        "b" + DegreesRoman.SIXTH.major,
        DegreesQualities.MAJOR_SEVENTH
    )
    SEVENTH = (
        AllChords.DOMINANT_SEVENTH,
        "b" + DegreesRoman.SEVENTH.major,
        DegreesQualities.DOMINANT_SEVENTH
    )


class MinorHarmonicSeventhDegrees(Degrees, Enum):
    FIRST = (
        AllChords.MINOR_MAJOR_SEVENTH,
        DegreesRoman.FIRST.minor + "♮7",
        DegreesQualities.MINOR_MAJOR_SEVENTH
    )
    SECOND = (
        AllChords.HALF_DIMINISHED_SEVENTH,
        DegreesRoman.SECOND.minor,
        DegreesQualities.HALF_DIMINISHED_SEVENTH
    )
    THIRD = (
        AllChords.AUGMENTED_MAJOR_SEVENTH,
        DegreesRoman.THIRD.major,
        DegreesQualities.AUGMENTED_MAJOR_SEVENTH
    )
    FOURTH = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.FOURTH.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    FIFTH = (
        AllChords.DOMINANT_SEVENTH,
        DegreesRoman.FIFTH.major,
        DegreesQualities.DOMINANT_SEVENTH
    )
    SIXTH = (
        AllChords.MAJOR_SEVENTH,
        DegreesRoman.SIXTH.major,
        DegreesQualities.MAJOR_SEVENTH
    )
    SEVENTH = (
        AllChords.DIMINISHED_SEVENTH,
        DegreesRoman.SEVENTH.minor,
        DegreesQualities.DIMINISHED_SEVENTH
    )


class MinorMelodicSeventhDegrees(Degrees, Enum):
    FIRST = (
        AllChords.MINOR_MAJOR_SEVENTH,
        DegreesRoman.FIRST.minor,
        DegreesQualities.MINOR_MAJOR_SEVENTH
    )
    SECOND = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.SECOND.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    THIRD = (
        AllChords.AUGMENTED_MAJOR_SEVENTH,
        "b" + DegreesRoman.THIRD.major,
        DegreesQualities.AUGMENTED_MAJOR_SEVENTH
    )
    FOURTH = (
        AllChords.DOMINANT_SEVENTH,
        DegreesRoman.FOURTH.major,
        DegreesQualities.DOMINANT_SEVENTH
    )
    FIFTH = (
        AllChords.DOMINANT_SEVENTH,
        DegreesRoman.FIFTH.major,
        DegreesQualities.DOMINANT_SEVENTH
    )
    SIXTH = (
        AllChords.HALF_DIMINISHED_SEVENTH,
        DegreesRoman.SIXTH.minor,
        DegreesQualities.HALF_DIMINISHED_SEVENTH
    )
    SEVENTH = (
        AllChords.HALF_DIMINISHED_SEVENTH,
        DegreesRoman.SEVENTH.minor,
        DegreesQualities.HALF_DIMINISHED_SEVENTH
    )


class DorianSeventhDegrees(Degrees, Enum):
    FIRST = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.FIRST.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    SECOND = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.SECOND.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    THIRD = (
        AllChords.MAJOR_SEVENTH,
        DegreesRoman.THIRD.major,
        DegreesQualities.MAJOR_SEVENTH
    )
    FOURTH = (
        AllChords.DOMINANT_SEVENTH,
        DegreesRoman.FOURTH.major,
        DegreesQualities.DOMINANT_SEVENTH
    )
    FIFTH = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.FIFTH.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    SIXTH = (
        AllChords.HALF_DIMINISHED_SEVENTH,
        DegreesRoman.SIXTH.minor,
        DegreesQualities.HALF_DIMINISHED_SEVENTH
    )
    SEVENTH = (
        AllChords.MAJOR_SEVENTH,
        DegreesRoman.SEVENTH.major,
        DegreesQualities.MAJOR_SEVENTH
    )


class PhrygianSeventhDegrees(Degrees, Enum):
    FIRST = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.FIRST.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    SECOND = (
        AllChords.MAJOR_SEVENTH,
        DegreesRoman.SECOND.major,
        DegreesQualities.MAJOR_SEVENTH
    )
    THIRD = (
        AllChords.DOMINANT_SEVENTH,
        DegreesRoman.THIRD.major,
        DegreesQualities.DOMINANT_SEVENTH
    )
    FOURTH = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.FOURTH.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    FIFTH = (
        AllChords.HALF_DIMINISHED_SEVENTH,
        DegreesRoman.FIFTH.minor,
        DegreesQualities.HALF_DIMINISHED_SEVENTH
    )
    SIXTH = (
        AllChords.MAJOR_SEVENTH,
        DegreesRoman.SIXTH.major,
        DegreesQualities.MAJOR_SEVENTH
    )
    SEVENTH = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.SEVENTH.minor,
        DegreesQualities.MINOR_SEVENTH
    )


class LydianSeventhDegrees(Degrees, Enum):
    FIRST = (
        AllChords.MAJOR_SEVENTH,
        DegreesRoman.FIRST.major,
        DegreesQualities.MAJOR_SEVENTH
    )
    SECOND = (
        AllChords.DOMINANT_SEVENTH,
        DegreesRoman.SECOND.major,
        DegreesQualities.DOMINANT_SEVENTH
    )
    THIRD = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.THIRD.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    FOURTH = (
        AllChords.HALF_DIMINISHED_SEVENTH,
        DegreesRoman.FOURTH.minor,
        DegreesQualities.HALF_DIMINISHED_SEVENTH
    )
    FIFTH = (
        AllChords.MAJOR_SEVENTH,
        DegreesRoman.FIFTH.major,
        DegreesQualities.MAJOR_SEVENTH
    )
    SIXTH = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.SIXTH.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    SEVENTH = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.SEVENTH.minor,
        DegreesQualities.MINOR_SEVENTH
    )


class MixolydianSeventhDegrees(Degrees, Enum):
    FIRST = (
        AllChords.DOMINANT_SEVENTH,
        DegreesRoman.FIRST.major,
        DegreesQualities.DOMINANT_SEVENTH
    )
    SECOND = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.SECOND.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    THIRD = (
        AllChords.HALF_DIMINISHED_SEVENTH,
        DegreesRoman.THIRD.minor,
        DegreesQualities.HALF_DIMINISHED_SEVENTH
    )
    FOURTH = (
        AllChords.MAJOR_SEVENTH,
        DegreesRoman.FOURTH.major,
        DegreesQualities.MAJOR_SEVENTH
    )
    FIFTH = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.FIFTH.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    SIXTH = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.SIXTH.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    SEVENTH = (
        AllChords.MAJOR_SEVENTH,
        DegreesRoman.SEVENTH.major,
        DegreesQualities.MAJOR_SEVENTH
    )


class LocrianSeventhDegrees(Degrees, Enum):
    FIRST = (
        AllChords.HALF_DIMINISHED_SEVENTH,
        DegreesRoman.FIRST.minor,
        DegreesQualities.HALF_DIMINISHED_SEVENTH
    )
    SECOND = (
        AllChords.MAJOR_SEVENTH,
        DegreesRoman.SECOND.major,
        DegreesQualities.MAJOR_SEVENTH
    )
    THIRD = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.THIRD.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    FOURTH = (
        AllChords.MINOR_SEVENTH,
        DegreesRoman.FOURTH.minor,
        DegreesQualities.MINOR_SEVENTH
    )
    FIFTH = (
        AllChords.MAJOR_SEVENTH,
        DegreesRoman.FIFTH.major,
        DegreesQualities.MAJOR_SEVENTH
    )
    SIXTH = (
        AllChords.DOMINANT_SEVENTH,
        DegreesRoman.SIXTH.major,
        DegreesQualities.DOMINANT_SEVENTH
    )
    SEVENTH = (
        AllChords.MAJOR_SEVENTH,
        DegreesRoman.SEVENTH.major,
        DegreesQualities.MAJOR_SEVENTH
    )


class SeventhsModes(Enum):
    MAJOR = [el for el in MajorSeventhDegrees]
    MINOR_NATURAL = [el for el in MinorNaturalSeventhDegrees]
    MINOR_HARMONIC = [el for el in MinorHarmonicSeventhDegrees]
    MINOR_MELODIC = [el for el in MinorMelodicSeventhDegrees]
    # Greek modes
    IONIAN = MAJOR
    DORIAN = [el for el in DorianSeventhDegrees]
    PHRYGIAN = [el for el in PhrygianSeventhDegrees]
    LYDIAN = [el for el in LydianSeventhDegrees]
    MIXOLYDIAN = [el for el in MixolydianSeventhDegrees]
    AEOLIAN = MINOR_NATURAL
    LOCRIAN = [el for el in LocrianSeventhDegrees]

    def degree(self, degree_index: int) -> str:
        return self.value[degree_index + 1][1]


# ===================9ths========================
class MajorNinthDegrees(Enum):
    pass
class MinorNaturalNinthDegrees(Enum):
    pass
class MinorHarmonicNinthDegrees(Enum):
    pass
class MinorMelodicNinthDegrees(Enum):
    pass
class DorianNinthDegrees(Enum):
    pass
class PhrygianNinthDegrees(Enum):
    pass
class LydianNinthDegrees(Enum):
    pass
class MixolydianNinthDegrees(Enum):
    pass
class LocrianNinthDegrees(Enum):
    pass


class NinthsModes(Enum):
    MAJOR = [el for el in MajorNinthDegrees]
    MINOR_NATURAL = [el for el in MinorNaturalNinthDegrees]
    MINOR_HARMONIC = [el for el in MinorHarmonicNinthDegrees]
    MINOR_MELODIC = [el for el in MinorMelodicNinthDegrees]
    # Greek modes
    IONIAN = MAJOR
    DORIAN = [el for el in DorianNinthDegrees]
    PHRYGIAN = [el for el in PhrygianNinthDegrees]
    LYDIAN = [el for el in LydianNinthDegrees]
    MIXOLYDIAN = [el for el in MixolydianNinthDegrees]
    AEOLIAN = MINOR_NATURAL
    LOCRIAN = [el for el in LocrianNinthDegrees]

    def degree(self, degree_index: int) -> str:
        return self.value[degree_index + 1][1]


class AccidentalNotes(Enum):
    """Altered notes in order of accidentals number in major
    and minor natural modes."""
    SHARPS = [
        NoteClassBase.F_SHARP,
        NoteClassBase.C_SHARP,
        NoteClassBase.G_SHARP,
        NoteClassBase.D_SHARP,
        NoteClassBase.A_SHARP,
        NoteClassBase.E_SHARP,
        NoteClassBase.B_SHARP,
    ]
    FLATS = [
        NoteClassBase.B_FLAT,
        NoteClassBase.E_FLAT,
        NoteClassBase.A_FLAT,
        NoteClassBase.D_FLAT,
        NoteClassBase.G_FLAT,
        NoteClassBase.C_FLAT,
        NoteClassBase.F_FLAT,
    ]


class AccidentalDegrees(Enum):
    """Altered degrees in minor and greek modes derived from the major mode.

    (Altered Degrees, Accidental Type)
    """
    # Alter notes in minor natural scale and add these accidentals
    MINOR_HARMONIC = (
        [DegreesRoman.SEVENTH],
        AccidentalsNames.SHARP
    )
    MINOR_MELODIC = (
        [DegreesRoman.SIXTH, DegreesRoman.SEVENTH],
        AccidentalsNames.SHARP
    )
    # Alter notes in major scale and add these accidentals
    DORIAN = (
        [DegreesRoman.THIRD, DegreesRoman.SEVENTH],
        AccidentalsNames.FLAT
    )
    PHRYGIAN = (
        [DegreesRoman.SECOND, DegreesRoman.THIRD, DegreesRoman.SIXTH, DegreesRoman.SEVENTH],
        AccidentalsNames.FLAT
    )
    LYDIAN = (
        [DegreesRoman.FOURTH],
        AccidentalsNames.SHARP
    )
    MIXOLYDIAN = (
        [DegreesRoman.SEVENTH],
        AccidentalsNames.FLAT
    )
    LOCRIAN = (
        [DegreesRoman.SECOND, DegreesRoman.THIRD, DegreesRoman.FIFTH, DegreesRoman.SIXTH, DegreesRoman.SEVENTH],
        AccidentalsNames.FLAT
    )

    @property
    def degrees(self) -> List[DegreesRoman]:
        return self.value[0]

    @property
    def symbol_accidentals(self) -> AccidentalsNames:
        return self.value[1]


class ModeConstructors(Enum):
    # Major modes
    MAJOR = (
        None,
        TriadsModes.MAJOR,
        SeventhsModes.MAJOR
    )
    DORIAN = (
        AccidentalDegrees.DORIAN,
        TriadsModes.DORIAN,
        SeventhsModes.DORIAN
    )
    PHRYGIAN = (
        AccidentalDegrees.PHRYGIAN,
        TriadsModes.PHRYGIAN,
        SeventhsModes.PHRYGIAN
    )
    LYDIAN = (
        AccidentalDegrees.LYDIAN,
        TriadsModes.LYDIAN,
        SeventhsModes.LYDIAN
    )
    MIXOLYDIAN = (
        AccidentalDegrees.MIXOLYDIAN,
        TriadsModes.MIXOLYDIAN,
        SeventhsModes.MIXOLYDIAN
    )
    LOCRIAN = (
        AccidentalDegrees.LOCRIAN,
        TriadsModes.LOCRIAN,
        SeventhsModes.LOCRIAN
    )
    # Minor modes
    NATURAL = (
        None,
        TriadsModes.MINOR_NATURAL,
        SeventhsModes.MINOR_NATURAL
    )
    HARMONIC = (
        AccidentalDegrees.MINOR_HARMONIC,
        TriadsModes.MINOR_HARMONIC,
        SeventhsModes.MINOR_HARMONIC
    )
    MELODIC = (
        AccidentalDegrees.MINOR_MELODIC,
        TriadsModes.MINOR_MELODIC,
        SeventhsModes.MINOR_MELODIC
    )

    def __repr__(self):
        return "<%s.%s>" % (self.__class__.__name__, self.name)

    @property
    def accidentals(self) -> Optional[AccidentalDegrees]:
        return self.value[0]

    @property
    def triads(self) -> TriadsModes:
        return self.value[1]

    @property
    def sevenths(self) -> SeventhsModes:
        return self.value[2]
    
    @property
    def ninths(self) -> NinthsModes:
        return self.value[3]

class Scales(Enum):
    MAJOR = {
        "MAJOR": ModeConstructors.MAJOR,
        "LYDIAN": ModeConstructors.LYDIAN,
        "MIXOLYDIAN": ModeConstructors.MIXOLYDIAN,
        "IONIAN": ModeConstructors.MAJOR,
    }
    MINOR = {
        "NATURAL": ModeConstructors.NATURAL,
        "HARMONIC": ModeConstructors.HARMONIC,
        "MELODIC": ModeConstructors.MELODIC,
        "DORIAN": ModeConstructors.DORIAN,
        "PHRYGIAN": ModeConstructors.PHRYGIAN,
        "LOCRIAN": ModeConstructors.LOCRIAN,
        "AEOLIAN": ModeConstructors.NATURAL,
    }


# Greek scales heritance the accidentals from major mode.
# This tonalities haver more than 7 accidentals in the major mode
# so we won't get the greek modes for them
NON_EXISTING_SCALES = [
    "G_SHARP_MINOR",
    "D_SHARP_MINOR",
    "A_SHARP_MINOR",
]


class Tonality(Enum):
    """
    Args:
        Enum ([type]): (
            Root note,
            Num Accidentals,
            Accidental object,
            {Additional Accidentals, Triads, Sevenths}
        )
    """
    # ----------- 0#, 0b -----------
    C_MAJOR = (
        NoteClassBase.C,
        0,
        None,
        Scales.MAJOR,
    )
    A_MINOR = (
        NoteClassBase.A,
        0,
        None,
        Scales.MINOR,
    )
    # ----------- 1# ---------------
    G_MAJOR = (
        NoteClassBase.G,
        1,
        AccidentalsNames.SHARP,
        Scales.MAJOR
    )
    E_MINOR = (
        NoteClassBase.E,
        1,
        AccidentalsNames.SHARP,
        Scales.MINOR,
    )
    # ----------- 1b ---------------
    F_MAJOR = (
        NoteClassBase.F,
        1,
        AccidentalsNames.FLAT,
        Scales.MAJOR
    )
    D_MINOR = (
        NoteClassBase.D,
        1,
        AccidentalsNames.FLAT,
        Scales.MINOR,
    )
    # ----------- 2# ---------------
    D_MAJOR = (
        NoteClassBase.D,
        2,
        AccidentalsNames.SHARP,
        Scales.MAJOR
    )
    B_MINOR = (
        NoteClassBase.B,
        2,
        AccidentalsNames.SHARP,
        Scales.MINOR,
    )
    # ----------- 2b ---------------
    B_FLAT_MAJOR = (
        NoteClassBase.B,
        2,
        AccidentalsNames.FLAT,
        Scales.MAJOR
    )
    G_MINOR = (
        NoteClassBase.G,
        2,
        AccidentalsNames.FLAT,
        Scales.MINOR,
    )
    # ----------- 3# ---------------
    A_MAJOR = (
        NoteClassBase.A,
        3,
        AccidentalsNames.SHARP,
        Scales.MAJOR
    )
    F_SHARP_MINOR = (
        NoteClassBase.F,
        3,
        AccidentalsNames.SHARP,
        Scales.MINOR,
    )
    # ----------- 3b ---------------
    E_FLAT_MAJOR = (
        NoteClassBase.E,
        3,
        AccidentalsNames.FLAT,
        Scales.MAJOR
    )
    C_MINOR = (
        NoteClassBase.C,
        3,
        AccidentalsNames.FLAT,
        Scales.MINOR,
    )
    # ----------- 4# ---------------
    E_MAJOR = (
        NoteClassBase.E,
        4,
        AccidentalsNames.SHARP,
        Scales.MAJOR
    )
    C_SHARP_MINOR = (
        NoteClassBase.C,
        4,
        AccidentalsNames.SHARP,
        Scales.MINOR,
    )
    # ----------- 4b ---------------
    A_FLAT_MAJOR = (
        NoteClassBase.A,
        4,
        AccidentalsNames.FLAT,
        Scales.MAJOR
    )
    F_MINOR = (
        NoteClassBase.F,
        4,
        AccidentalsNames.FLAT,
        Scales.MINOR,
    )
    # ----------- 5# ---------------
    B_MAJOR = (
        NoteClassBase.B,
        5,
        AccidentalsNames.SHARP,
        Scales.MAJOR
    )
    G_SHARP_MINOR = (
        NoteClassBase.G,
        5,
        AccidentalsNames.SHARP,
        Scales.MINOR,
    )
    # ----------- 5b ---------------
    D_FLAT_MAJOR = (
        NoteClassBase.D,
        5,
        AccidentalsNames.FLAT,
        Scales.MAJOR
    )
    B_FLAT_MINOR = (
        NoteClassBase.B,
        5,
        AccidentalsNames.FLAT,
        Scales.MINOR,
    )
    # ----------- 6# ---------------
    F_SHARP_MAJOR = (
        NoteClassBase.F,
        6,
        AccidentalsNames.SHARP,
        Scales.MAJOR
    )
    D_SHARP_MINOR = (
        NoteClassBase.D,
        6,
        AccidentalsNames.SHARP,
        Scales.MINOR,
    )
    # ----------- 6b ---------------
    G_FLAT_MAJOR = (
        NoteClassBase.G,
        6,
        AccidentalsNames.FLAT,
        Scales.MAJOR
    )
    E_FLAT_MINOR = (
        NoteClassBase.E,
        6,
        AccidentalsNames.FLAT,
        Scales.MINOR,
    )
    # ----------- 7# ---------------
    C_SHARP_MAJOR = (
        NoteClassBase.C,
        7,
        AccidentalsNames.SHARP,
        Scales.MAJOR
    )
    A_SHARP_MINOR = (
        NoteClassBase.A,
        7,
        AccidentalsNames.SHARP,
        Scales.MINOR,
    )
    # ----------- 7b ---------------
    C_FLAT_MAJOR = (
        NoteClassBase.C,
        7,
        AccidentalsNames.FLAT,
        Scales.MAJOR
    )
    A_FLAT_MINOR = (
        NoteClassBase.A,
        7,
        AccidentalsNames.FLAT,
        Scales.MINOR,
    )

    @property
    def relative(self) -> Tonality:

        for tonality in Tonality.__members__.values():
            if self.value[1] == tonality.value[1] and self.value[2] == tonality.value[2] and self.value[3] != tonality.value[3]:
                return tonality

    @property
    def root_note(self) -> int:
        return self.value[0]

    @property
    def num_accidentals(self) -> AccidentalsNames:
        return self.value[1]

    @property
    def symbol_accidentals(self) -> AccidentalsNames:
        return self.value[2]

    @property
    def major(self) -> ModeConstructors:
        return self.scales("MAJOR")

    @property
    def dorian(self) -> ModeConstructors:
        if self.name not in NON_EXISTING_SCALES:
            return self.scales("DORIAN")

    @property
    def phrygian(self) -> ModeConstructors:
        if self.name not in NON_EXISTING_SCALES:
            return self.scales("PHRYGIAN")

    @property
    def lydian(self) -> ModeConstructors:
        return self.scales("LYDIAN")

    @property
    def mixolydian(self) -> ModeConstructors:
        return self.scales("MIXOLYDIAN")

    @property
    def locrian(self) -> ModeConstructors:
        if self.name not in NON_EXISTING_SCALES:
            return self.scales("LOCRIAN")

    @property
    def natural(self) -> ModeConstructors:
        return self.scales("NATURAL")

    @property
    def harmonic(self) -> ModeConstructors:
        return self.scales("HARMONIC")

    @property
    def melodic(self) -> ModeConstructors:
        return self.scales("MELODIC")

    def scales(self, scale: str) -> Optional[ModeConstructors]:
        if scale in self.value[3].value.keys():
            return self.value[3].value[scale]
        else:
            return None

    @property
    def all_scales(self) -> AccidentalsNames:
        all_scales = []
        if "MAJOR" in self.name:
            all_scales.append(self.major)
            all_scales.append(self.lydian)
            all_scales.append(self.mixolydian)
        elif "MINOR" in self.name:
            all_scales.append(self.natural)
            all_scales.append(self.harmonic)
            all_scales.append(self.melodic)
            if self.name not in NON_EXISTING_SCALES:
                all_scales.append(self.dorian)
                all_scales.append(self.phrygian)
                all_scales.append(self.locrian)
        return all_scales

    @property
    def altered_notes(self) -> List[NoteClassBase]:
        """Returns the altered notes in the scale.
        This method only returns the altered nottes for the major and minor
        natural modes which are the "parent" modes.
        To obtain the additional accidentals of the submodes (minor harmonic...),
        use the `submode_altered_notes` method."""
        if self.symbol_accidentals == AccidentalsNames.FLAT:
            notes = [note for note in AccidentalNotes.FLATS.value[:self.num_accidentals]]
        elif self.symbol_accidentals == AccidentalsNames.SHARP:
            notes = [note for note in AccidentalNotes.SHARPS.value[:self.num_accidentals]]
        else:
            notes = []
        return notes

    def scale_notes(self, scale: str) -> List[NoteClassBase]:
        """This method returns the notes of the scale corresponding to
        a submode.
        This is only used in the case of minor scales (harmonic or melodic) and greek scales.
        The values that support the scales arg are: :func:`~musicaiz.harmony.Scales`.

        Examples
        --------
        Major tonalities:

        >>> tonality = Tonality.D_MAJOR
        >>> tonality.scale_notes("MAJOR")
        >>> tonality.scale_notes("LYDIAN")
        >>> tonality.scale_notes("LYDIAN")
        >>> tonality.scale_notes("MIXOLYDIAN")
        >>> tonality.scale_notes("IONIAN")

        Minor tonalities:

        >>> tonality = Tonality.C_MINOR
        >>> tonality.scale_notes("NATURAL")
        >>> tonality.scale_notes("HARMONIC")
        >>> tonality.scale_notes("DORIAN")
        >>> tonality.scale_notes("PHRYGIAN")
        >>> tonality.scale_notes("LOCRIAN")
        >>> tonality.scale_notes("AEOLIAN")
        """
        # Obtain altered notes depending on the scale (minor harmonic...)
        if isinstance(scale, str):
            scale_inst = self.scales(scale)  # initialize scale
        else:
            scale_inst = scale
        more_accidentals = scale_inst.accidentals
        if more_accidentals is None:
            return self.notes
        altered_degrees = more_accidentals.degrees
        symbol = more_accidentals.symbol_accidentals
        # All the greek scales inherit the accidentals from major mode
        # but the melodic and harmonic scales have the accidentals of the minor natural mode
        if ("MINOR" in self.name) and (scale != "NATURAL" and scale != "MELODIC" and scale != "HARMONIC"):
            # count "_" in tonality name
            if self.name.count("_") == 1:
                major_tonality = self.name.split("_")[0] + "_" + "MAJOR"
            elif self.name.count("_") == 2:
                major_tonality = self.name.split("_")[0] + "_" + self.name.split("_")[1] + "_" + "MAJOR"
            notes = Tonality[major_tonality].notes
        else:
            notes = self.notes
        for degree in altered_degrees:
            # Obtain the note that corresponds to the degree
            note = notes[degree.index]
            if symbol == AccidentalsNames.SHARP:
                notes[degree.index] = note.add_sharp
            elif symbol == AccidentalsNames.FLAT:
                notes[degree.index] = note.add_flat
        return notes

    @property
    def notes(self) -> List[NoteClassBase]:
        """Uses `altered_notes` property to generate the notes in the scale."""
        # list of the 7 notes in natural scale
        notes = [note for note in NoteClassBase if note.natural_scale_index is not None]
        # sort the list of the notes in natural scale from root note
        for _ in range(self.root_note.natural_scale_index):
            notes.append(notes.pop(0))
        # Now we alter the notes with accidentals depending on the scale and mode
        scale_notes = copy.deepcopy(notes)
        for idx, note in enumerate(scale_notes):
            for altered_note in self.altered_notes:
                natural_note = NoteClassBase.get_natural_note(altered_note)
                if natural_note == note:
                    scale_notes[idx] = altered_note
        return scale_notes

    def __repr__(self):
        return "<%s.%s>" % (self.__class__.__name__, self.name)

    # TODO
    @classmethod
    def get_scale_from_accidentals_num(cls) -> Scales:
        pass

    # TODO
    @classmethod
    def get_scale_from_accidentals_symbol(cls) -> Scales:
        pass

    # TODO
    @classmethod
    def get_scale_from_accidentals_num_symbol(cls) -> Scales:
        pass

    @classmethod
    def get_chord_from_degree(
        cls,
        tonality: str,
        degree: str,
        scale: Optional[Union[str, ModeConstructors]] = None,
        chord_type: str = "triad",
    ) -> Tuple(NoteClassBase, AllChords):
        # TODO: Add Non valid scale exception
        # TODO: Add Non valid degree exception
        # TODO: Add Non valid scale for mode exception
        # TODO: Add non valid chord type exception
        # Get the notes in the scale
        if not isinstance(tonality, str):
            tonality = tonality.name
        # Get scale inside the tonality
        scale_mode = cls._get_scale(tonality, scale)
        print(scale, scale_mode, tonality)
        if scale is None:
            notes = Tonality[tonality].notes
        else:
            notes = Tonality[tonality].scale_notes(scale_mode.name)
        # Get degree index in the scale
        if isinstance(degree, str):
            # Get degree obj
            degree_obj = DegreesRoman.get_name_with_degree(degree)
            degree_idx = degree_obj.index
        elif isinstance(degree, int):
            degree_idx = degree

        if chord_type == "triad":
            chord = scale_mode.triads.value[degree_idx].chord
        elif chord_type == "seventh":
            chord = scale_mode.sevenths.value[degree_idx].chord
        else:
            raise ValueError(f"Input chord type {chord_type} not known.")
        # Get root note
        root_note = notes[degree_idx]
        return root_note, chord

    @staticmethod
    def _get_scale(
        tonality: str,
        scale: Optional[Union[str, ModeConstructors]] = None,
    ) -> ModeConstructors:
        """For major mode the default is the major scale. For
        the minor mode the default is the minor natural scale."""
        tonality_inst = Tonality[tonality]
        if scale is not None:
            if isinstance(scale, str):
                scale_mode = tonality_inst.scales(scale)
            else:
                scale_mode = scale
        else:
            if "MAJOR" in tonality:
                scale_mode = tonality_inst.scales("MAJOR")
            elif "MINOR" in tonality:
                scale_mode = tonality_inst.scales("NATURAL")
        return scale_mode

    @classmethod
    def get_all_chords_from_scale(
        cls,
        tonality: str,
        scale: Union[str, ModeConstructors] = None,
        chord_type: str = "triad",
    ) -> List[Tuple[NoteClassBase, AllChords]]:
        chords = []
        scale = cls._get_scale(tonality, scale)
        for i in range(7):
            chords.append(cls.get_chord_from_degree(tonality, i, scale, chord_type))
        return chords

    @classmethod
    def get_chord_notes_from_degree(
        cls,
        tonality: str,
        degree: str,
        scale: Union[str, ModeConstructors] = None,
        chord_type: str = "triad",
        inversion: int = 0,
    ) -> List[NoteClassBase]:
        """Return a list of the notes that corresponds to the input degree chord."""
        # TODO: Take into account inversion and chord_type
        # Get the notes in the scale
        root_note, chord = cls.get_chord_from_degree(tonality, degree, scale, chord_type)
        # Get intervals from root note
        intervals = chord.intervals
        notes = [root_note]
        for interval in intervals:
            interval_inst = Interval(interval)
            # Initialize note with name and octave
            # TODO: Maybe this will go in a helper method in intervals module
            degree_root = root_note.value[0].contracted + "1"
            note_obj = Interval._initialize_note(degree_root)
            note_dest_obj = interval_inst.transpose_note(note_obj)
            notes.append(note_dest_obj.note)
        return notes

    @classmethod
    def get_scales_degrees_from_chord(
        cls,
        chord: Tuple[NoteClassBase, AllChords]
    ) -> List[Tuple[DegreesRoman, Tonality, ModeConstructors]]:
        """Returns all the scales with their corresponding degree
        that belong to the input chord."""
        # Go through all the tonalities
        scales_degrees = []
        for tonality in cls.__members__.values():
            # Loop through all the scales in each tonality
            for scale in tonality.all_scales:
                chords_scale = cls.get_all_chords_from_scale(tonality=tonality.name, scale=scale.name)
                # Go through all the degrees chords in the scale
                # TODO: Take into account 7ths
                for degree_idx, degree_chord in enumerate(chords_scale):
                    if set(chord) == set(degree_chord):
                        degree_obj = DegreesRoman.get_degree_from_index(degree_idx)
                        scales_degrees.append((degree_obj, tonality, scale))
                        # TODO: break this loop and go to scales loop, there're not 2 same chords in a scale
        return scales_degrees

    @classmethod
    def get_modes_degrees_from_chord(
        cls,
        chord: Tuple[NoteClassBase, AllChords]
    ) -> List[Tuple[DegreesRoman, Tonality]]:
        """Same as `get_scales_degrees_from_chord` but this method only returns
        the tonality and degree, not the specific scale."""
        all_degrees = cls.get_scales_degrees_from_chord(chord)
        prev_degree = (0, 0, 0)
        degree_tonalities = []
        for degree in all_degrees:
            if degree[0] != prev_degree[0] and degree[1] != prev_degree[1]:
                degree_tonalities.append((degree[0], degree[1]))
            prev_degree = degree
        return degree_tonalities
