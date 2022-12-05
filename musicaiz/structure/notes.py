from __future__ import annotations
import pretty_midi as pm
from typing import List, Tuple, Union, Optional, Dict
import re
from enum import Enum

# Our modules
from musicaiz.rhythm import (
    TimingConsts,
    ms_per_tick,
    get_symbolic_duration,
    SymbolicNoteLengths,
    Timing
)


class AccidentalsNames(Enum):
    SHARP = ["#", "sharp", "sostenido"]
    FLAT = ["b", "flat", "bemol"]
    NATURAL = ["♮", "natural", "becuadro"]

    @property
    def contracted(self) -> str:
        return self.value[0]

    @property
    def expanded(self) -> str:
        return self.value[1]

    @property
    def spanish(self) -> str:
        return self.value[2]


# TODO: This might be necessary in future methods/classes
class AccidentalsValues(Enum):
    """The number of semitones that are added to a note if
    the accidental is present.
    Note that the `natural` accidental will cancel the effect of other accidental
    but only one (if the note has 2 sharps, the `natural` will only
    cancel the effect of one sharp)."""
    SHARP = 1
    FLAT = -1
    NATURAL = 0


class NotesMidiOctaves(Enum):
    MIN_OCTAVE = -1
    MAX_OCTAVE = 9


class NoteClassNames(Enum):

    """
    This class contains all the possible notes with their names

    Examples
    --------
    >>> NoteClassNames.check_note_name_exists("C")
    """
    C = ["C", "Do"]
    C_SHARP = ["C#", "Do #", "C sharp", "Do sostenido"]
    C_FLAT = ["Cb", "Do b", "C flat", "Do bemol"]
    D = ["D", "Re"]
    D_SHARP = ["D#", "Re #", "D sharp", "Re sostenido"]
    D_FLAT = ["Db", "Re b", "D flat", "Re bemol"]
    E = ["E", "Mi"]
    E_SHARP = ["E#", "Mi #", "E sharp", "Mi sostenido"]
    E_FLAT = ["Eb", "Mi b", "E flat", "Mi bemol"]
    F = ["F", "Fa"]
    F_SHARP = ["F#", "Fa #", "F sharp", "Fa sostenido"]
    F_FLAT = ["Fb", "Fa b", "F flat", "Fa bemol"]
    G = ["G", "Sol"]
    G_SHARP = ["G#", "Sol #", "G sharp", "Sol sostenido"]
    G_FLAT = ["Gb", "Sol b", "G flat", "Sol bemol"]
    A = ["A", "La"]
    A_SHARP = ["A#", "La #", "A sharp", "La sostenido"]
    A_FLAT = ["Ab", "La b", "A flat", "La bemol"]
    B = ["B", "Si"]
    B_SHARP = ["B#", "Si #", "B sharp", "Si sostenido"]
    B_FLAT = ["Bb", "Si b", "B flat", "Si bemol"]

    @property
    def spanish_contracted(self) -> str:
        return self.value[1]

    @property
    def contracted(self) -> str:
        return self.value[0]

    @property
    def expanded(self) -> str:
        if len(self.value) <= 2:
            return self.value[0]
        else:
            return self.value[2]

    @property
    def spanish_expanded(self) -> str:
        if len(self.value) <= 2:
            return self.value[1]
        else:
            return self.value[3]

    @classmethod
    def get_all_names(cls) -> List[str]:
        all_notes = []
        for note in cls.__members__.values():
            for n in note.value:
                all_notes.append(n)
        return all_notes

    @classmethod
    def check_note_name_exists(cls, name: str) -> bool:
        all_notes = NoteClassNames.get_all_names()
        return name in all_notes

    @classmethod
    def get_note_with_name(cls, note_name: str) -> NoteClassNames:
        for note in cls.__members__.values():
            for n in note.value:
                if note_name == n:
                    return note


class NoteClassBase(Enum):
    """
    The Tuples have 3 values:
        Value 1: the most common and abbreviate note name
        Value 2: index of note in natural scale (no accidentals)
            This index is the one that we'll use to calculate the interval class (2nd, 3rd...)
        Value 3: index of the note in chromatic scale
            This value will give us the distance in semitones between 2 notes.
    """
    C = (NoteClassNames.C, 0, 0)
    B_SHARP = (NoteClassNames.B_SHARP, None, 0)
    C_SHARP = (NoteClassNames.C_SHARP, None, 1)
    D_FLAT = (NoteClassNames.D_FLAT, None, 1)
    D = (NoteClassNames.D, 1, 2)
    D_SHARP = (NoteClassNames.D_SHARP, None, 3)
    E_FLAT = (NoteClassNames.E_FLAT, None, 3)
    E = (NoteClassNames.E, 2, 4)
    F_FLAT = (NoteClassNames.F_FLAT, None, 4)
    E_SHARP = (NoteClassNames.E_SHARP, None, 5)
    F = (NoteClassNames.F, 3, 5)
    F_SHARP = (NoteClassNames.F_SHARP, None, 6)
    G_FLAT = (NoteClassNames.G_FLAT, None, 6)
    G = (NoteClassNames.G, 4, 7)
    G_SHARP = (NoteClassNames.G_SHARP, None, 8)
    A_FLAT = (NoteClassNames.A_FLAT, None, 8)
    A = (NoteClassNames.A, 5, 9)
    A_SHARP = (NoteClassNames.A_SHARP, None, 10)
    B_FLAT = (NoteClassNames.B_FLAT, None, 10)
    B = (NoteClassNames.B, 6, 11)
    C_FLAT = (NoteClassNames.C_FLAT, None, 11)

    def __repr__(self):
        return "<%s.%s>" % (self.__class__.__name__, self.name)

    @property
    def natural_scale_index(self) -> int:
        return self.value[1]

    @property
    def chromatic_scale_index(self) -> int:
        return self.value[2]

    @property
    def add_sharp(self) -> NoteClassBase:
        new_semitones = self.value[2] + AccidentalsValues.SHARP.value
        return NoteClassBase._get_note_from_chromatic_idx(new_semitones)[0]

    @property
    def add_flat(self) -> NoteClassNames:
        new_semitones = self.value[2] - AccidentalsValues.SHARP.value
        return NoteClassBase._get_note_from_chromatic_idx(new_semitones)[0]

    @staticmethod
    def get_natural_note(note: NoteClassBase) -> NoteClassNames:
        """Returns the natural note corresponding to an altered note."""
        # if note name has _ it's an altered note
        if "_" in note.name:
            natural_note_name = note.name.split("_")[0]
            return NoteClassBase[natural_note_name]
        else:
            return note

    @staticmethod
    def _get_note_from_chromatic_idx(semitones: int) -> NoteClassNames:
        """Look for the note idx in the chromatic scale"""
        notes = []
        semitones = semitones % 12
        for i in NoteClassBase.__members__.values():
            if i.value[2] == semitones:
                notes.append(i)
        return notes

    @staticmethod
    def all_chromatic_scale_indexes() -> List[Tuple[str, int]]:
        return [note.value[2] for note in NoteClassBase]

    @staticmethod
    def all_natural_scale_indexes() -> List[Tuple[str, int]]:
        return [note.value[1] for note in NoteClassBase]

    @classmethod
    def get_note_with_name(cls, note_name: str) -> NoteClassBase:
        note_class = NoteClassNames.get_note_with_name(note_name)
        for note in cls.__members__.values():
            if note_class == note.value[0]:
                return note

    @classmethod
    def get_natural_scale_notes(cls) -> List[NoteClassBase]:
        notes = []
        for note in cls.__members__:
            if cls[str(note)].natural_scale_index is not None:
                notes.append(cls[str(note)])
        return notes

    @classmethod
    def get_all_chromatic_scale_notes(cls) -> List[NoteClassBase]:
        """Returns all the notes in chromatic scale (flats AND sharps)."""
        notes = [cls[str(note)] for note in cls.__members__]
        return notes

    @classmethod
    def get_notes_chromatic_scale(cls, alteration: str = "FLAT") -> List[NoteClassBase]:
        notes = cls.get_all_chromatic_scale_notes()
        if alteration == "FLAT":
            contrary_alt = "SHARP"
        elif alteration == "SHARP":
            contrary_alt = "FLAT"

        for note in cls.get_all_chromatic_scale_notes():
            if alteration == "SHARP":
                if (alteration in note.name) and ("B_" in note.name or "E_" in note.name):
                    notes.remove(note)
            elif alteration == "FLAT":
                if (alteration in note.name) and ("F_" in note.name or "C_" in note.name):
                    notes.remove(note)
            if contrary_alt in note.name:
                notes.remove(note)
        return notes


class NoteValue:

    """
    This class allows to instantiate a note object by just giving the pitch
    value or the name of the note in the MIDI format.

    Parameters
    ----------

    Raises
    ------
    ValueError
        if the input pitch value is out of the range 0-127 (MIDI format).

    ValueError
        if the input note name does not exists in the MIDI format.

    Examples
    --------

    >>> note_1 = musanalysis.NoteValue("C4")
    >>> note_2 = musanalysis.NoteValue(120)
    """

    def __init__(self, pitch: Union[str, int]):

        if isinstance(pitch, int):
            if self.pitch_value_in_range(pitch):
                self.pitch_name = pm.note_number_to_name(pitch)
                self.pitch = pitch
                self.note_name, self.octave = self.split_pitch_name(self.pitch_name)
                self.note = NoteClassBase.get_note_with_name(self.note_name)
            else:
                raise ValueError(f"Introduced pitch value {pitch} is not valid.")

        elif isinstance(pitch, str):
            try:
                # Get octave and pitch name
                self.note_name, self.octave = self.split_pitch_name(pitch)
            except:
                raise ValueError(f"Introduced note {pitch} is not in a valid format.")

            if NotesMidiOctaves.MIN_OCTAVE.value <= int(self.octave) <= NotesMidiOctaves.MAX_OCTAVE.value:
                # the pitch name is in the midi range (ex: `C#50` does not exist)
                self.pitch = pm.note_name_to_number(pitch)
                self.pitch_name = pitch
                self.note_name, self.octave = self.split_pitch_name(self.pitch_name)
                self.note = NoteClassBase.get_note_with_name(self.note_name)
            else:
                raise ValueError("Note octave is not in a valid MIDI range.")

        chromatic_index = self.note.chromatic_scale_index
        note_enharmonics = NoteClassBase._get_note_from_chromatic_idx(chromatic_index)
        if len(note_enharmonics) == 1:
            self.enharmonic = note_enharmonics[0]
        else:
            self.enharmonic = [note for note in note_enharmonics if note != self.note][0]

    @staticmethod
    def split_pitch_name(pitch_name: str) -> Tuple[str, str]:
        """Splits a note name (`C#1` or `Cb-1`...) into a note name (`C` or `Cb` ...)
        and the octave (`1` or `1` ...)"""

        note = [i for i in re.split(r'([A-Za-z#]+)', pitch_name) if i]
        note_name = note[0]
        octave = note[1]

        return note_name, octave

    @staticmethod
    def split_note_name(note_name: str) -> Tuple[str, str]:
        """Splits a note name (`C#` or `Cb`...) into a note name (`C`)
        and the accidental (`#` or `b` ...)"""

        note = [i for i in re.split(r'([A-Z]+)', note_name) if i]
        note_natural = note[0]
        accidental = None
        if len(note) > 1:
            accidental = note[1]

        return note_natural, accidental

    @staticmethod
    def pitch_value_in_range(pitch_value: int) -> bool:
        if pitch_value >= 0 and pitch_value <= 127:
            return True
        else:
            return False

    def __repr__(self):
        return "Note(pitch={}, name={}, octave={})".format(
            self.pitch,
            self.note_name,
            self.octave
        )


# TODO: Maybe here instead of passing int for ticks or float for sec which can be tricky,
# we should write an additional arg called `unit` = "secs" or "ticks" so then
# and then initialize the note just passing start and end and being the time
# unit the one specified in `unit` argument. (Then, check that ticks are int and
# secs are float will always be mandatory).
class NoteTiming(NoteValue):

    def __init__(
        self,
        pitch: Union[str, int],
        start: Union[int, float],
        end: Union[int, float],
        bpm: int = TimingConsts.DEFAULT_BPM.value,
        resolution: int = TimingConsts.RESOLUTION.value,
    ):

        """

        Attributes
        ----------
        pitch: int

        start: Union[str, int]
            the note on. If the input argument is an int we consider
            that the value correspond to ticks unit, and if the
            argument is a float we assume the value corresponds to seconds.

        end: Union[int, float]
            the note off. If the input argument is an int we consider
            that the value correspond to ticks unit, and if the
            argument is a float we assume the value corresponds to seconds.

        delta_time: int
            the note duration in 16th notes.

        bpm: int
        the tempo or bpms.

        resolution: int
            the pulses o ticks per quarter note (PPQ or TPQN).

        Raises:
            ValueError: [description]
            ValueError: [description]
        """

        # TODO: What if we include `delta_time` as an input arg, and if it's not None
        # we calculate the start_sec, end_sec, start_tick and end_ticks automatically?

        super().__init__(pitch)

        self.ms_tick = ms_per_tick(bpm, resolution)

        timings = Timing._initialize_timing_attributes(start, end, self.ms_tick)

        self.start_ticks = timings["start_ticks"]
        self.end_ticks = timings["end_ticks"]
        self.start_sec = timings["start_sec"]
        self.end_sec = timings["end_sec"]

        self.symbolic = get_symbolic_duration(
            self.end_ticks - self.start_ticks,
            True,
            resolution
        )

    def __repr__(self):
        return "Note(pitch={}, " \
               "name={}, " \
               "start_sec={}, " \
               "end_sec={}, " \
               "start_ticks={}, " \
               "end_ticks={}, " \
               "symbolic={})".format(
                   self.pitch,
                   self.note_name,
                   self.start_sec,
                   self.end_sec,
                   self.start_ticks,
                   self.end_ticks,
                   SymbolicNoteLengths[self.symbolic].value)


class Note(NoteTiming):

    """A note even with its relevant attributes.

    Initializing:

    - Time: A note can be initialized in seconds with the note start `note_on` and
            note end `note_off` attributes or in ticks with `start_ticks` and
            `duration_ticks`.

            If note on, note off, start_icks and end_ticks are given, the object will
            be initialized with the `note_on` and `note_off` values ignoring the
            `start_ticks` and ènd_ticks` input values.

    - Pitch: A note can be initialized by giving the name of the pitch: `C0`...or by
            giving the pitch value: 12...

            If the pitch value is provided, the note name is ignored if it is also provided.
            This is done for preventing non valid pitch name and values pairs.

    Params:
        pitch: [description]
        note_on: [description]
        note_off: [description]
        velocity: [description]
        start_ticks: [description]
        duration_ticks: [description]
    """

    __slots__ = [
        "note",
        "pitch",
        "pitch_name",
        "note_name",
        "octave",
        "enharmonic",
        "start_sec",
        "end_sec",
        "start_ticks",
        "end_ticks",
        "symbolic",
        "velocity",
        "bpm",
        "resolution",
        "ligated",
        "instrument_prog",
        "bar_idx",
        "instrument_idx",
        "is_drum"
    ]

    def __init__(
        self,
        pitch: Union[str, int],
        start: Union[int, float],
        end: Union[int, float],
        velocity: int,
        instrument_prog: Optional[int] = None,
        instrument_idx: Optional[int] = None,
        bar_idx: Optional[int] = None,
        beat_idx: Optional[int] = None,
        subbeat_idx: Optional[int] = None,
        ligated: bool = False,
        bpm: int = TimingConsts.DEFAULT_BPM.value,
        resolution: int = TimingConsts.RESOLUTION.value,
        is_drum: bool = False
    ):

        super().__init__(pitch, start, end, bpm, resolution)

        self.velocity = velocity
        self.instrument_prog = instrument_prog
        self.bar_idx = bar_idx
        self.beat_idx = beat_idx
        self.subbeat_idx = subbeat_idx

        # We can have 2 instruments (or tracks) with the same program number,
        # so this will store the index of the instrument to distinguish equal
        # program number instruments in MIDI files
        self.instrument_idx = instrument_idx

        # if a note belongs to 2 bars and we split the tracks by bars
        self.ligated = ligated

        self.resolution = resolution
        self.bpm = bpm
        self.is_drum = is_drum

    def __repr__(self):
        return "Note(pitch={}, " \
               "name={}, " \
               "start_sec={:f}, " \
               "end_sec={:f}, " \
               "start_ticks={}, " \
               "end_ticks={}, " \
               "symbolic={}, " \
               "velocity={}, " \
               "ligated={}, " \
               "instrument_prog={}, " \
               "bar_idx={}, " \
               "beat_idx={}, " \
               "subbeat_idx={})".format(
                   self.pitch,
                   self.note_name,
                   self.start_sec,
                   self.end_sec,
                   self.start_ticks,
                   self.end_ticks,
                   SymbolicNoteLengths[self.symbolic].value,
                   self.velocity,
                   self.ligated,
                   self.instrument_prog,
                   self.bar_idx,
                   self.beat_idx,
                   self.subbeat_idx,
                )
