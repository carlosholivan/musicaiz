from typing import List, Union, Optional
import numpy as np


from musicaiz.rhythm import (
    TimingConsts,
    ms_per_bar,
    ms_per_tick,
    Timing,
)
from musicaiz.structure import Note


class Bar:

    """Defines a class to group notes in bars.

    Attributes
    ----------

    time_sig: str
        If we do know the time signature in advance, we can initialize Musa object with it.
        This will assume that all the MIDI has the same time signature.

    bpm: int
        The tempo or bpm of the MIDI file. If this parameter is not initialized we suppose
        120bpm with a resolution (sequencer ticks) of 960 ticks, which means that we have
        500 ticks per quarter note.

    resolution: int
        the pulses o ticks per quarter note (PPQ or TPQN). If this parameter is not initialized
        we suppose a resolution (sequencer ticks) of 960 ticks.

    absolute_timing: bool
        selects how note timing attributes are initialized when reading a MIDI file.
        If `absolute_timing` is True, notes will be written in absolute times.
        If `absolute_timing` is False, times will be relative to the bar start.
    """

    def __init__(
        self,
        start: Optional[Union[int, float]] = None,
        end: Optional[Union[int, float]] = None,
        time_sig: str = TimingConsts.DEFAULT_TIME_SIGNATURE.value,
        bpm: int = TimingConsts.DEFAULT_BPM.value,
        resolution: int = TimingConsts.RESOLUTION.value,
        absolute_timing: bool = True,
    ):
        self.bpm = bpm
        self.time_sig = time_sig
        self.resolution = resolution
        self.absolute_timing = absolute_timing

        # The following attributes are set when loading a MIDI file
        # with Musa class
        self.note_density = None
        self.harmonic_density = None

        self.ms_tick = ms_per_tick(bpm, resolution)

        if start is not None and end is not None:
            timings = Timing._initialize_timing_attributes(start, end, self.ms_tick)

            self.start_ticks = timings["start_ticks"]
            self.end_ticks = timings["end_ticks"]
            self.start_sec = timings["start_sec"]
            self.end_sec = timings["end_sec"]
        else:
            self.start_ticks = None
            self.end_ticks = None
            self.start_sec = None
            self.end_sec = None

    def relative_notes_timing(self, bar_start: float):
        """The bar start is the value in ticks where the bar starts"""
        ms_tick = ms_per_tick(self.bpm, self.resolution)
        for note in self.notes:
            note.start_ticks = note.start_ticks - bar_start
            note.end_ticks = note.end_ticks - bar_start
            note.start_sec = note.start_ticks * ms_tick / 1000
            note.end_sec = note.end_ticks * ms_tick / 1000

    @staticmethod
    def get_last_note(note_seq: List[Note]) -> float:
        """Get last note in note_seq."""
        end_secs = 0
        for note in note_seq:
            if note.end_sec > end_secs:
                last_note = note
        return last_note

    def get_bars_durations(
        self,
        note_seq: List[Note]
    ) -> List[float]:
        """
        Build array of bar durations.
        We suppose that the note_seq is in the same time signature.
        """
        last_note = self.get_last_note(note_seq)
        end_secs = last_note.end_secs
        sec_measure = ms_per_bar(self.time_sig, self.bpm) * 1000
        bar_durations = np.arange(0, end_secs, sec_measure).tolist()
        if end_secs % sec_measure != 0:
            bar_durations.append(bar_durations[-1] + sec_measure)
        return bar_durations

    @classmethod
    def get_total_bars(cls, note_seq: List[Note]) -> int:
        return len(cls.get_bars_durations(note_seq))

    @classmethod
    def group_notes_in_bars(cls, note_seq: List[Note]) -> List[List[Note]]:
        bars_durations = cls.get_bars_durations(note_seq)
        bars = []
        prev_bar_sec = 0
        for bar_sec in bars_durations:
            for note in note_seq:
                if bar_sec >= note.end_sec and prev_bar_sec < note.end_sec:
                    bars.append(note)
            prev_bar_sec = bar_sec
        return bars

    def __repr__(self):
        if self.start_sec is not None:
            start_sec = round(self.start_sec, 2)
        else:
            start_sec = self.start_sec
        if self.end_sec is not None:
            end_sec = round(self.end_sec, 2)
        else:
            end_sec = self.end_sec

        return "Bar(time_signature={}, " \
                "note_density={}, " \
                "harmonic_density={} " \
                "start_ticks={} " \
                "end_ticks={} " \
                "start_sec={} " \
                "end_sec={})".format(
                    self.time_sig,
                    self.note_density,
                    self.harmonic_density,
                    self.start_ticks,
                    self.end_ticks,
                    start_sec,
                    end_sec
                )
