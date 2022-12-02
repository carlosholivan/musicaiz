from __future__ import annotations
from abc import ABCMeta
from enum import Enum
from typing import Tuple, List, Dict, Union, Optional
import numpy as np
import math


class TimingConsts(Enum):
    """
    This Enum contains the default values for timing parameters.

    RESOLUTION: Sequencer Ticks per quarter note (TPQN) or
    Pulses per quarter note (PPQ) used in Logic Pro.

    DEFAULT_BPM: Default Tempo or bpm if no bpm value is known in advance.

    DEFAULT_TIME_SIGNATURE: Default time signature if it  is not known in advance.
    """
    RESOLUTION = 96
    TICKS_PER_SEC = 192
    DEFAULT_BPM = 120
    DEFAULT_TIME_SIGNATURE = "4/4"


class NoteLengths(Enum):
    """
    This Enum contains Note durations in fractions of the whole note.
    The Enum names correspond to the American notation.
    From https://www.fransabsil.nl/htm/ticktime.htm

    Value: Relative value
    """
    DOTTED_WHOLE = 3 / 2
    WHOLE = 1
    DOTTED_HALF = 3 / 4
    HALF = 1 / 2
    DOTTED_QUARTER = 3 / 8
    QUARTER = 1 / 4
    DOTTED_EIGHT = 3 / 16
    EIGHT = 1 / 8
    DOTTED_SIXTEENTH = 3 / 32
    SIXTEENTH = 1 / 16
    DOTTED_THIRTY_SECOND = 3 / 64
    THIRTY_SECOND = 1 / 32
    DOTTED_SIXTY_FOUR = 3 / 128
    SIXTY_FOUR = 1 / 64
    DOTTED_HUNDRED_TWENTY_EIGHT = 3 / 256
    HUNDRED_TWENTY_EIGHT = 1 / 128

    # triplets
    HALF_TRIPLET = 1 / 3
    QUARTER_TRIPLET = 1 / 6
    EIGHT_TRIPLET = 1 / 12
    SIXTEENTH_TRIPLET = 1 / 24


    @property
    def fraction(self) -> float:
        return self.value

    def ticks(
        self,
        resolution: int = TimingConsts.RESOLUTION.value,
    ) -> int:
        return round(resolution * self.fraction / NoteLengths.QUARTER.fraction)

    def ms(
        self,
        bpm: int = TimingConsts.DEFAULT_BPM.value,
        resolution: int = TimingConsts.RESOLUTION.value,
    ) -> float:
        return ms_per_tick(bpm, resolution) * self.ticks(resolution)

    @classmethod
    def get_note_ticks_mapping(
        cls,
        triplets: bool = False,
        resolution: int = TimingConsts.RESOLUTION.value,
    ) -> Dict[str, int]:
        dict_notes = {}
        for note_dur in list(cls.__members__.keys()):
            if not triplets:
            # remove triplet durations (optional)
                if "TRIPLET" in note_dur:
                    continue
            dict_notes.update({note_dur: cls[note_dur].ticks(resolution)})
        return dict_notes

    @classmethod
    def get_note_with_fraction(cls, fraction: float) -> NoteLengths:
        for note in cls.__members__:
            if math.isclose(round(cls[note].value, 3), round(fraction, 3)):
                return cls[note]


class SymbolicNoteLengths(Enum):
    """
    This Enum contains Note symbols as unicode strings.
    """
    DOTTED_WHOLE = u"\U0001D15D" + "."
    WHOLE = u"\U0001D15D"
    DOTTED_HALF = u"\U0001D15E" + "."
    HALF = u"\U0001D15E"
    DOTTED_QUARTER = u"\U0001D15F" + "."
    QUARTER = u"\U0001D15F"
    DOTTED_EIGHT = u"\U0001D160" + "."
    EIGHT = u"\U0001D160"
    DOTTED_SIXTEENTH = u"\U0001D161" + "."
    SIXTEENTH = u"\U0001D161"
    DOTTED_THIRTY_SECOND = u"\U0001D162" + "."
    THIRTY_SECOND = u"\U0001D162"
    DOTTED_SIXTY_FOUR = u"\U0001D163" + "."
    SIXTY_FOUR = u"\U0001D163"
    DOTTED_HUNDRED_TWENTY_EIGHT = u"\U0001D164" + "."
    HUNDRED_TWENTY_EIGHT = u"\U0001D164"

    # triplets
    HALF_TRIPLET = u"\U0001D15E\U000000B3"
    QUARTER_TRIPLET = u"\U0001D15F\U000000B3"
    EIGHT_TRIPLET = u"\U0001D160\U000000B3"
    SIXTEENTH_TRIPLET = u"\U0001D161\U000000B3"


class TimeSigDenominators(Enum):
    """This Enum contains the possible time signature denominators."""
    WHOLE = 1
    HALF = 2
    QUARTER = 4
    EIGHT = 8
    SIXTEENTH = 16
    THIRTY_SECOND = 32
    SIXTY_FOUR = 64
    HUNDRED_TWENTY_EIGHTH = 128

    @classmethod
    def get_note_length(cls, denominator: int) -> TimeSigDenominators:
        got = None
        for note in cls.__members__.values():
            if note.value == denominator:
                got = note
                return got
        if got is None:
            raise ValueError(f"Not note found for {denominator} denominator.")


class TimeSignature:

    def __init__(self, time_sig: Union[Tuple[int, int], str]):
        """
        TimeSignature class for representing time signatures.

        Parameters
        ----------
            time_sig: str
                The time signature in the format num/den
        """
        if isinstance(time_sig, str):
            if "/" not in time_sig:
                raise ValueError(f"Time signature {time_sig} is not in the correct format.")
            self.time_sig = time_sig
            self.num, self.denom = self.time_sig.split("/")
            self.num = int(self.num)
            self.denom = int(self.denom)
        elif isinstance(time_sig, tuple):
            self.time_sig = str(time_sig[0]) + "/" + str(time_sig[1])
            self.num = int(time_sig[0])
            self.denom = int(time_sig[1])
        else:
            raise ValueError(f"Time signature {time_sig} is not in the correct format.")

    @property
    def beats_per_bar(self) -> int:
        return self.num

    @property
    def beat_type(self) -> str:
        return TimeSigDenominators.get_note_length(self.denom).name

    def _notes_per_bar(self, note_name: str) -> int:
        return (1 / NoteLengths[note_name].value) * self.num * (1 / self.denom)

    @property
    def quarters(self) -> int:
        # Get the name of the denominator note: NoteLengths(1 / self.denom).name
        return self._notes_per_bar("QUARTER")

    @property
    def eights(self) -> int:
        return self._notes_per_bar("EIGHT")

    @property
    def sixteenths(self) -> int:
        return self._notes_per_bar("SIXTEENTH")

    def __repr__(self):
        return "TimeSig(num={}, den={})".format(
            self.num,
            self.denom,
        )


class Timing(metaclass=ABCMeta):

    def __init__(
        self,
        bpm: float,
        resolution: int,
        start: Union[int, float],
        end: Union[int, float],
    ):
        self.ms_tick = ms_per_tick(bpm, resolution)

        timings = self._initialize_timing_attributes(
            start, end, self.ms_tick
        )

        self.start_ticks = timings["start_ticks"]
        self.end_ticks = timings["end_ticks"]
        self.start_sec = timings["start_sec"]
        self.end_sec = timings["end_sec"]
        self.bpm = bpm
        self.resolution = resolution

    @staticmethod
    def _initialize_timing_attributes(
        start: Union[int, float],
        end: Union[int, float],
        ms_tick: Union[int, float],
    ) -> Dict[str, Union[int, float]]:
        # inital checks
        if start < 0 or end <= 0:
            raise ValueError("Start and end must be positive.")
        elif start >= end:
            raise ValueError("Start time must be lower than the end time.")

        # ticks must be int, secs must be float
        if isinstance(start, int) and isinstance(end, int):
            start_ticks = start
            end_ticks = end
            start_sec = start_ticks * ms_tick / 1000
            end_sec = end_ticks * ms_tick / 1000
        elif isinstance(start, float) and isinstance(end, float):
            start_sec = start
            end_sec = end
            start_ticks = int(start_sec * (1 / (ms_tick / 1000)))
            end_ticks = int(end_sec * (1 / (ms_tick / 1000)))

        timings = {
            "start_ticks": start_ticks,
            "end_ticks": end_ticks,
            "start_sec": start_sec,
            "end_sec": end_sec,
        }
        return timings


class Beat(Timing):

    def __init__(
        self,
        bpm: float,
        resolution: int,
        start: Union[int, float],
        end: Union[int, float],
        time_sig: Optional[TimeSignature] = None,
        global_idx: Optional[int] = None,
        bar_idx: Optional[int] = None,
    ):

        super().__init__(bpm, resolution, start, end)

        self.time_sig = time_sig
        self.global_idx = global_idx
        self.bar_idx = bar_idx
        self.symbolic = TimeSigDenominators.get_note_length(
            time_sig.denom
        ).name.lower()

    def __repr__(self):

        return "Beat(time_signature={}, " \
                "bpm={}, " \
                "start_ticks={} " \
                "end_ticks={} " \
                "start_sec={} " \
                "end_sec={} " \
                "global_idx={} " \
                "bar_idx={} " \
                "symbolic={})".format(
                    self.time_sig,
                    self.bpm,
                    self.start_ticks,
                    self.end_ticks,
                    self.start_sec,
                    self.end_sec,
                    self.global_idx,
                    self.bar_idx,
                    self.symbolic,
                )


class Subdivision(Timing):

    def __init__(
        self,
        bpm: float,
        resolution: int,
        start: Union[int, float],
        end: Union[int, float],
        time_sig: Optional[TimeSignature] = None,
        global_idx: Optional[int] = None,
        bar_idx: Optional[int] = None,
        beat_idx: Optional[int] = None,
    ):

        super().__init__(bpm, resolution, start, end)

        self.time_sig = time_sig
        self.global_idx = global_idx
        self.bar_idx = bar_idx
        self.beat_idx = beat_idx

    def __repr__(self):

        return "Subdivision(time_signature={}, " \
                "bpm={}, " \
                "start_ticks={} " \
                "end_ticks={} " \
                "start_sec={} " \
                "end_sec={} " \
                "global_idx={} " \
                "bar_idx={} " \
                "beat_idx={})".format(
                    self.time_sig,
                    self.bpm,
                    self.start_ticks,
                    self.end_ticks,
                    self.start_sec,
                    self.end_sec,
                    self.global_idx,
                    self.bar_idx,
                    self.beat_idx,
                )


def ms_per_tick(
    bpm: int = TimingConsts.DEFAULT_BPM.value,
    resolution: int = TimingConsts.RESOLUTION.value,
) -> float:
    """This function calculates the miliseconds that correspond to one tick.

    Parameters
    ----------

    bpm: int
        the tempo or bpms.

    resolution: int
        the pulses o ticks per quarter note (PPQ or TPQN).

    Returns
    -------

    float
        The miliseconds that correspond to a tick.
    """
    return 60000 / (bpm * resolution)


def ms_per_note(
    note_length: str = "quarter",
    bpm: int = TimingConsts.DEFAULT_BPM.value,
    resolution: int = TimingConsts.RESOLUTION.value,
) -> float:
    """This function calculates the miliseconds that correspond to one note.

    Parameters
    ----------

    note_length: str
        the name of the note length in american notation.

    bpm: int
        the tempo or bpms.

    Returns
    -------

    float
        The miliseconds that correspond to a note length.
    """

    return NoteLengths[note_length.upper()].ms(bpm, resolution)


def ticks_per_bar(
    time_sig: str = TimingConsts.DEFAULT_TIME_SIGNATURE.value,
    resolution: int = TimingConsts.RESOLUTION.value,
) -> Tuple[int, int]:
    """This function calculates the ticks that correspond to one beat and bar.

    Returns
    -------

    ticks_beat: int
        number of ticks in a beat of the bar.

    ticks_bar: int
        number of ticks in a bar.
    """

    # There are measures with beats or subdivisions lower or higher than a quarter note.
    # The number of beats per measure is given by the 1st value of the `measure` tuple.
    # The beat figure (quarter note, 16th note...) is given by the 2nd value of the `measure` tuple.
    n, d = _bar_str_to_tuple(time_sig)
    # Find the note corresponding note to the value of the time sig. denominator
    note_length = TimeSigDenominators.get_note_length(d)
    # That note is the beat, so we get now the ticks in that beat
    ticks_beat = NoteLengths[note_length.name].ticks(resolution)
    # The numerator gives us how many beats are in the bar
    ticks_bar = n * ticks_beat
    return ticks_beat, ticks_bar


def ms_per_bar(
    time_sig: str = TimingConsts.DEFAULT_TIME_SIGNATURE.value,
    bpm: int = TimingConsts.DEFAULT_BPM.value,
    resolution: int = TimingConsts.RESOLUTION.value,
) -> float:
    """This function calculates the miliseconds that correspond to one bar.

    Parameters
    ----------

    time_sig: str
        the time signature as a fraction.

    bpm: int
        the tempo or bpms.

    resolution: int
        the pulses o ticks per quarter note (PPQ or TPQN).

    Returns
    -------

    ms_beat: float
        number of miliseconds in a beat of the bar.

    ms_bar: float
        number of miliseconds in a bar.
    """

    ticks_beat, ticks_bar = ticks_per_bar(time_sig, resolution)
    ms_tick = ms_per_tick(bpm, resolution)
    ms_beat, ms_bar = ticks_beat * ms_tick, ticks_bar * ms_tick
    return ms_beat, ms_bar


def _bar_str_to_tuple(
    time_sig: str = "4/4",
) -> Tuple[int, int]:
    """This function converts the time signature str to a tuple of
    2 values: numerator and denominator."""
    if "/" not in time_sig:
        raise ValueError("Not valid time signature format. Time sig must have '/'")
    numerator, denominator = time_sig.split("/")
    return int(numerator), int(denominator)


def get_subdivisions(
    total_bars: int,
    subdivision: str,  # TODO: give possible values here
    time_sig: str = TimingConsts.DEFAULT_TIME_SIGNATURE.value,
    bpm: int = TimingConsts.DEFAULT_BPM.value,
    resolution: int = TimingConsts.RESOLUTION.value,
    absolute_timing: bool = True
) -> List[Dict[str, Union[int, float]]]:

    """
    This method returns the grid (vertical lines) for quantizing notes
    of a determined number of measures given by `total_bars`.
    The lines are plotted in each beat subdivision.
    Each line has the format m.b.s.:
    m: measure
    b: beat
    s: subdivision
    Ex.: subdivision = 8 is an 8th note

    Parameters
    ----------

    total_bars: int
        the number of bars.

    subdivision: str
        the note subdivision. Ex.: quarter, eight...
        Note that we cannot have a subdivision greater than the beat nor a
        non value of the beat note. If the beat corresponds to a quarter note
        (X/4 time sigs.), the subdivision has to be a note shorter than the quarter note
        but half or 4 times less than the beat (eight, sixteenth note...).

    time_sig: str
        the time signature as a fraction.

    bpm: int
        the tempo or bpms.

    resolution: int
        the pulses o ticks per quarter note (PPQ or TPQN).

    absolute_timing: bool
        default is True. This allows to initialize note time arguments in absolute (True) or
        relative time units (False). Relative units means that each bar will start at 0 seconds
        and ticks, so the note timing attributes will be relative to the bar start equals to 0.

    Returns
    -------

    beat_subdivs: List[Dict[str, Union[int, float]]]
        each element in the list is a subdivision. The subdivision dict key's are:

        `bar`: the bar index in the total_bars of the subdivision.

        `piece_beat`: the beat index corresponding to the subdivision in the total_bars or piece.

        `piece_subdivision`: the subdivision index corresponding to the total number of
        subdivisions in the total_bars or piece.

        `bar_beat`: the beat index corresponding to the subdivision in its bar.

        `bar_subdivision`: the subdivision index corresponding to its bar.

        `ticks`: the ticks value of where the subdivision starts.

        `sec`: the secs value of where the subdivision starts.
    """

    if total_bars <= 0 or isinstance(total_bars, float):
        raise ValueError("Total number of bars must be a positive integer.")

    beats_bar, denominator = _bar_str_to_tuple(time_sig)

    # Get the note length
    note_length = TimeSigDenominators.get_note_length(denominator)

    # Get the ticks in a beat
    ticks_beat = NoteLengths[note_length.name].ticks(resolution)
    ms_beat = ms_per_note(note_length.name.lower(), bpm)

    # Get the subdivision note
    subdiv_note = TimeSigDenominators[subdivision.upper()]

    # Subdivision note has to be 2, 4, 8...times less than the beat
    # Subdivision must be in the subdivisions possible values (strs)
    if subdivision.upper() not in TimeSigDenominators.__members__.keys():
        raise ValueError(f"Subdivision {subdivision} does not exist.")
    if subdiv_note.value % note_length.value != 0:
        raise ValueError(f"Input subdivision {subdivision} is not valid for beat note {note_length.name}.")

    # Get how many of these subdivision notes are in a beat
    subdivs_beat = subdiv_note.value // note_length.value
    subdivs_bar = beats_bar * subdivs_beat
    ticks_subdiv = ticks_beat // subdivs_beat
    ms_subdiv = ms_beat / subdivs_beat

    # total beats in the grid (vertical lines) are the measures * number of beats per measure
    total_beats = total_bars * beats_bar
    total_beat_subdivs = total_beats * subdivs_beat

    # Calculate the ticks where each subdivision starts.
    # That will be the horizontal position of the vertical or grid lines.
    beat_subdivs = []
    measure_index = 1
    beat_index = 1
    beat_bar_index = 1
    for beat_subdiv_index in range(total_beat_subdivs):
        beat_subdivision = {}
        measure_div = beat_subdiv_index % subdivs_bar
        beat_div = beat_subdiv_index % subdivs_beat
        if measure_div == 0 and not beat_subdiv_index == 0:
            measure_index += 1
            beat_bar_index = 0
        if beat_div == 0 and not beat_subdiv_index == 0:
            beat_index += 1
            beat_bar_index += 1

        beat_subdivision["bar"] = measure_index
        beat_subdivision["piece_beat"] = beat_index
        beat_subdivision["piece_subdivision"] = beat_subdiv_index + 1
        beat_subdivision["bar_beat"] = beat_bar_index
        beat_subdivision["bar_subdivision"] = measure_div + 1
        beat_subdivision["ticks"] = int(ticks_subdiv * beat_subdiv_index)
        beat_subdivision["sec"] = ms_subdiv * beat_subdiv_index / 1000
        beat_subdivs.append(beat_subdivision)

    # update ticks and sec values if absolute_timing is not True
    if not absolute_timing:
        # for the convention used, the 1st element in the dict `beat_subdivs`
        # has a "bar" value of 1 (not 0), so we'll start in the else statement
        prev_bar = 0
        for subdiv in beat_subdivs:
            if subdiv["bar"] == prev_bar:
                subdiv["ticks"] = subdiv["ticks"] - bar_start_ticks
                subdiv["sec"] = subdiv["sec"] - bar_start_sec
            else:
                bar_start_ticks, bar_start_sec = subdiv["ticks"], subdiv["sec"]
                subdiv["ticks"], subdiv["sec"] = 0, 0
                prev_bar = subdiv["bar"]

    return beat_subdivs


def get_symbolic_duration(
    duration: int,
    triplets: bool = False,
    resolution: int = TimingConsts.RESOLUTION.value,
) -> str:
    """Given a note duration in ticks it calculates its symbolic
    duration: half, quarter, dotted_half...

    Parameters
    -----------

    triplets: bool
        default is False. It takes (True) or not (False) into account
        triplets durations.
    """

    all_notes_ticks = NoteLengths.get_note_ticks_mapping(triplets, resolution)
    notes_ticks = all_notes_ticks

    # look for the closest note in the notes ticks dict
    # the closest note will be the selected note even if its
    # duration in ticks is not exactly equal to the theoretical note ticks
    arr = np.asarray(list(notes_ticks.values()))
    i = (np.abs(arr - duration)).argmin()
    symbolic_duration = list(notes_ticks.keys())[i]
    return symbolic_duration
