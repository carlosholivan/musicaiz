from typing import Tuple

from musicaiz.harmony import Tonality, AllChords, DegreesRoman
from musicaiz.structure import NoteClassBase


class BPSFH:

    @classmethod
    def bpsfh_key_to_musicaiz(cls, note: str) -> Tonality:
        alt = None
        if "-" in note:
            alt = "FLAT"
            note = note.split("-")[0]
        elif "+" in note:
            alt = "SHARP"
            note = note.split("+")[0]
        if note.isupper():
            mode = "MAJOR"
        else:
            mode = "MINOR"
            note = note.capitalize()
        if alt is None:
            tonality = Tonality[note + "_" + mode]
        else:
            tonality = Tonality[note + "_" + alt + "_" + mode]
        return tonality

    @classmethod
    def bpsfh_chord_quality_to_musicaiz(cls, quality: str) -> AllChords:
        if quality == "M":
            q = "MAJOR_TRIAD"
        elif quality == "m":
            q = "MINOR_TRIAD"
        elif quality == "M7":
            q = "MAJOR_SEVENTH"
        elif quality == "m7":
            q = "MINOR_SEVENTH"
        elif quality == "D7":
            q = "DOMINANT_SEVENTH"
        elif quality == "a":
            q = "AUGMENTED_TRIAD"
        return AllChords[q]

    @classmethod
    def bpsfh_chord_to_musicaiz(
        cls,
        note: str,
        degree: int,
        quality: str,
    ) -> Tuple[NoteClassBase, AllChords]:
        tonality = cls.bpsfh_key_to_musicaiz(note)
        qt = cls.bpsfh_chord_quality_to_musicaiz(quality)
        notes = tonality.notes
        return notes[degree-1], qt
