"""
Loaders
-------

musicaiz main class in called Musa. This class loads a file and
initializes all the attributes at different levels depending on the
input params.

.. autosummary::
    :toctree: generated/

    Musa
"""

from __future__ import annotations
from typing import TextIO, Union, List, Optional, Type
import pretty_midi as pm
from pathlib import Path
from enum import Enum
import copy


# Our modules
from musicaiz.structure import Note, Instrument, Bar
from musicaiz.rhythm import (
    TimingConsts,
    ms_per_bar,
    TimeSignature,
    Beat,
    Subdivision,
    advanced_quantizer,
    QuantizerConfig,
)
from musicaiz.converters import musa_to_prettymidi
from musicaiz.features import get_harmonic_density
from musicaiz.algorithms import key_detection, KeyDetectionAlgorithms


class ValidFiles(Enum):
    MIDI = [".mid", ".midi"]
    MUSIC_XML = [".xml"]

    @classmethod
    def all_extensions(cls) -> List[str]:
        all = []
        for i in cls.__members__.values():
            for n in i.value:
                all.append(n)
        return all


class Musa:

    __slots__ = [
        "file",
        "tonality",
        "time_signature_changes",
        "resolution",
        "instruments",
        "is_quantized",
        "total_bars",
        "absolute_timing",
        "cut_notes",
        "notes",
        "bars",
        "tempo_changes",
        "instruments_progs",
        "general_midi",
        "subdivision_note",
        "subbeats",
        "beats",
        "quantizer_args",
    ]

    # subdivision_note and quantize_note
    # lower than a quarter note which is a beat in X/4 bars
    VALID_SUBDIVISIONS = [
        "eight",
        "sixteenth",
        "thirty_two",
        "sixty_four",
        "hundred_twenty_eight",
    ]

    def __init__(
        self,
        file: Optional[Union[str, TextIO, Path]],
        quantize: bool = False,
        cut_notes: bool = False,
        tonality: Optional[str] = None,
        resolution: Optional[int] = None,
        absolute_timing: bool = True,
        general_midi: bool = False,
        subdivision_note: str = "sixteenth",
        quantizer_args: Type[QuantizerConfig] = QuantizerConfig,
    ):

        """
        Structure: attributes that contains lists of Note and Instrument objects.
        Time: attributes that contains lists of Bar, Beat and Subdivision
        objects.

        A MIDI file can contain time signature changes, so each Beat objects are equivalent
        to the Bar they belong to. Ex.: a 2/4 time signature will contain 2 beats = 2 quarter
        notes whereas a 3/8 bar will contain 3 beats = 3 eight notes.
        """

        # TODO: quantize
        # TODO: relative times in notes?
        # TODO: cut_notes
        # TODO: assign notes their name when key is known
        # TODO: key signature changes
        # TODO: write_midi -> with pretty_midi
        # TODO: synthesize -> with pretty_midi

        self.instruments = []
        self.notes = []
        self.total_bars = 0
        self.general_midi = general_midi
        self.absolute_timing = absolute_timing
        self.is_quantized = quantize
        self.subdivision_note = subdivision_note
        self.subbeats = []
        self.beats = []
        self.cut_notes = cut_notes
        self.time_signature_changes = []
        self.bars = []
        self.tempo_changes = []
        self.quantizer_args = quantizer_args

        if subdivision_note not in self.VALID_SUBDIVISIONS:
            raise ValueError(
                "{subdivision_note} is not valid subdivision_note. " \
                    "Valid values are: {self.VALID_SUBDIVISIONS}"
            )

        # File provided
        if file is not None:
            if isinstance(file, str):
                file = Path(file)
            if self.is_valid(file):
                self.file = file
            else:
                raise ValueError("Input file extension is not valid.")
            if self.is_midi(file):
                self._load_midifile(resolution, tonality)

    def json(self):
        return {key: getattr(self, key, None) for key in self.__slots__}

    def _load_midifile(
        self,
        resolution: int,
        tonality: str,
    ):
        # initialize midi object with pretty_midi
        pm_inst = pm.PrettyMIDI(
            midi_file=str(self.file),
        )
        if resolution is None:
            self.resolution = pm_inst.resolution
        else:
            self.resolution = resolution
        prev_time = -1
        for time_sig in pm_inst.time_signature_changes:
            # there cannot be 2 different time sigs at the same time
            if time_sig.time == prev_time:
                continue
            self.time_signature_changes.append(
                {
                    "time_sig": TimeSignature(
                        (time_sig.numerator, time_sig.denominator)
                    ),
                    "ms": time_sig.time * 1000,
                }
            )
            prev_time = time_sig.time

        # if the time signature is not defined, we'll initialize it as
        # 4/4 by default
        if len(self.time_signature_changes) == 0:
            self.time_signature_changes = [
                {
                    "time_sig": TimeSignature(
                        TimingConsts.DEFAULT_TIME_SIGNATURE.value
                    ),
                    "ms": 0.0
                }
            ]
        # delete time signature changes if the time signature is equal to the next one
        ts_changes = copy.deepcopy(self.time_signature_changes)
        for i in range(len(ts_changes) - 1):
            if ts_changes[i]["time_sig"].num == ts_changes[i+1]["time_sig"].num and \
            ts_changes[i]["time_sig"].denom == ts_changes[i+1]["time_sig"].denom:
                del self.time_signature_changes[i+1]

        # For whatever reason, get_tempo_changes() returns a tuple
        # of arrays with 2 elements per array, the 1st is the time and
        # the 2nd is the tempo. In other cases, only 2 arrays are returned
        # each one with one element, the 1st array is the time and the 2nd, the tempo
        tempo_changes = pm_inst.get_tempo_changes()
        if tempo_changes[0].size >= 2:
            for tc in zip(tempo_changes[0], tempo_changes[1]):
                self.tempo_changes.append(
                    {"tempo": tc[1], "ms": tc[0] * 1000}
                )
        elif tempo_changes[0].size == 1 and len(tempo_changes) == 2:
            self.tempo_changes.append(
                {"tempo": tempo_changes[1][0], "ms": tempo_changes[0][0] * 1000}
            )

        # Initialize tonality. If it's given, ignore the KeySignatureChanges
        if tonality is None:
            self.tonality = tonality
        else:
            self.tonality = pm_inst.key_signature_changes

        last_note_end = self._get_last_note_end(pm_inst)

        # Add last note end to the time signature changes
        # (for easier bars loading)
        self.tempo_changes.append(
            {"tempo": self.tempo_changes[-1]["tempo"], "ms": last_note_end * 1000}
        )
        # delete tempo changes if next one is at the same time
        #self.tempo_changes = [
        #    self.tempo_changes[i]
        #    for i in range(len(self.tempo_changes) - 1)
        #    if self.tempo_changes[i-1]["ms"] + 20 >= self.tempo_changes[i]["ms"]
        #]

        # Load beats
        self._load_beats(last_note_end)

        # Load subdivisions
        self._load_subdivisions(last_note_end)

        # group beats in bars and create Bar objects
        self._load_bars_and_group_beats_in_bars()

        # last bar is complete even if the last note does not
        # end when the bar ends, so we'll add the empty beats
        # to the last bar to complete the bar (as it's done in DAWs)
        self.total_bars = len(self.bars)

        # Map instruments to load them as musicaiz instrument class
        self._load_instruments_and_notes(pm_inst)
        self.instruments_progs = [inst.program for inst in self.instruments]

        # Fill bar attributes related to notes information
        self._fill_bar_notes_attributes()

        # assign bar indexes to subbeats
        j, k = 0, 0
        for _, subbeat in enumerate(self.subbeats):
            bar = self.bars[j]
            beat = self.beats[k]
            # bar_idx
            if subbeat.start_sec >= bar.start_sec and subbeat.end_sec <= bar.end_sec:
                subbeat.bar_idx = j
            else:
                j += 1
                subbeat.bar_idx = j
            # beat_idx
            if subbeat.start_sec >= beat.start_sec and subbeat.end_sec <= beat.end_sec:
                subbeat.beat_idx = k
            else:
                k += 1
                subbeat.beat_idx = k

        assert len([sub for sub in self.subbeats if sub.bar_idx is None]) == 0
        assert len([sub for sub in self.subbeats if sub.beat_idx is None]) == 0

        # Add subbeats to last bar if it's incomplete
        if self.subbeats[-1].end_ticks < self.bars[-1].end_ticks:
            subbeats_last_bar = self.get_subbeats_in_bar(len(self.bars) - 1)
            # subbeats in a complete bar
            subbeats_total = int(
                self.bars[-1].time_sig._notes_per_bar(self.subdivision_note.upper())
            )
            if len(subbeats_last_bar) < subbeats_total:
                for _ in range(subbeats_total - len(subbeats_last_bar)):
                    dur = subbeats_last_bar[0].end_sec - subbeats_last_bar[0].start_sec
                    subbeat = Subdivision(
                        time_sig=subbeats_last_bar[0].time_sig,
                        start=self.subbeats[-1].end_sec,
                        end=self.subbeats[-1].end_sec + dur,
                        bpm=subbeats_last_bar[0].bpm,
                        resolution=self.subbeats[-1].resolution,
                    )
                    subbeat.global_idx = len(self.subbeats)
                    subbeat.bar_idx = len(self.bars) - 1
                    # TODO: Group last subbeats in the correct beats
                    # (this is not that important since these beats do not contain notes)
                    subbeat.beat_idx = len(self.beats) - 1
                    subbeat.bar_idx = len(self.bars) - 1
                    self.subbeats.append(subbeat)

        # if quantize
        if self.is_quantized:
            quantized_notes = []
            for i, bar in enumerate(self.bars):
                v_grid = [sb.start_ticks for sb in self.get_subbeats_in_bar(i)]
                # TODO: Recalcualte subbeat_idx, beat_idx and bar_idx of the notes
                notes = self.get_notes_in_bar(i)
                advanced_quantizer(
                    notes, v_grid, config=self.quantizer_args,
                    bpm=bar.bpm, resolution=self.resolution
                )
                quantized_notes.extend(notes)
            self.notes = quantized_notes

    @classmethod
    def is_valid(cls, file: Union[str, Path]):
        extension = cls.get_file_extension(file)
        return True if extension in ValidFiles.all_extensions() else False

    @staticmethod
    def get_file_extension(file: Union[str, Path]):
        return Path(file).suffix

    @classmethod
    def is_midi(cls, file: Union[str, Path]):
        extension = cls.get_file_extension(file)
        return True if extension in ValidFiles.MIDI.value else False

    @classmethod
    def is_musicxml(cls, file: Union[str, TextIO]):
        extension = cls.get_file_extension(file)
        return True if extension in ValidFiles.MUSIC_XML.value else False

    def bar_beats_subdivs_analysis(self):
        for i, time_sig in enumerate(self.time_signature_changes):
            if i + 1 == len(self.time_signature_changes):
                break
            sb_len = len([sb for sb in self.subbeats if sb.time_sig.time_sig == time_sig["time_sig"].time_sig])
            print(f"{sb_len} subdivisions in {time_sig['time_sig'].time_sig}")
            beat_len = len([beat for beat in self.beats if beat.time_sig.time_sig == time_sig["time_sig"].time_sig])
            print(f"{beat_len} beats in {time_sig['time_sig'].time_sig}")
            bar_len = len([bar for bar in self.bars if bar.time_sig.time_sig == time_sig["time_sig"].time_sig])
            print(f"{bar_len} bars in {time_sig['time_sig'].time_sig}")

    # subbeat
    def get_notes_in_subbeat(
        self,
        subbeat_idx: int,
        program: Optional[Union[List[int], int]] = None,
        instrument_idx: Optional[Union[List[int], int]] = None,
    ) -> List[Note]:
        notes = self._filter_by_instruments(program, instrument_idx, self.notes)
        return self._get_objs_in_subbeat(subbeat_idx, notes)

    def get_notes_in_subbeat_bar(
        self,
        subbeat_idx: int,
        bar_idx: int,
        program: Optional[Union[List[int], int]] = None,
        instrument_idx: Optional[Union[List[int], int]] = None,
    ) -> List[Note]:
        first_idx = len(self.get_subbeats_in_bars(0, bar_idx))
        global_idx = subbeat_idx + first_idx
        all_notes = self.get_notes_in_bar(bar_idx, program, instrument_idx)
        return self._get_objs_in_subbeat(global_idx, all_notes)

    def _get_objs_in_subbeat(self, subbeat_idx: int, objs):
        if subbeat_idx >= len(self.subbeats):
            raise ValueError(
                f"Not subbeat index {subbeat_idx} found in bars. The file has {len(self.subbeats)} subbeats."
            )
        return list(filter(lambda obj: obj.subbeat_idx == subbeat_idx, objs))

    # subbeats
    def get_notes_in_subbeats(
        self,
        subbeat_start: int,
        subbeat_end: int,
        program: Optional[Union[List[int], int]] = None,
        instrument_idx: Optional[Union[List[int], int]] = None,
    ) -> List[Note]:
        notes = self._filter_by_instruments(program, instrument_idx, self.notes)
        return self._get_objs_in_subbeats(
            subbeat_start, subbeat_end, notes
        )

    def _get_objs_in_subbeats(
        self,
        subbeat_start: int,
        subbeat_end: int,
        objs: List[Note]
    ):
        if subbeat_start > subbeat_end:
            raise ValueError("subbeat_start must be minor than subbeat_end.")
        return list(
            filter(
                lambda obj: obj.subbeat_idx >= subbeat_start and obj.subbeat_idx < subbeat_end, objs
            )
        )

    # beat
    def get_notes_in_beat(
        self,
        beat_idx: int,
        program: Optional[Union[List[int], int]] = None,
        instrument_idx: Optional[Union[List[int], int]] = None,
    ) -> List[Note]:
        """beat_idx is the global index of the beat in the file."""
        notes = self._filter_by_instruments(program, instrument_idx, self.notes)
        return self._get_objs_in_beat(beat_idx, notes) if notes is not None else []

    def get_notes_in_beat_bar(
        self,
        beat_idx: int,
        bar_idx: int,
        program: Optional[Union[List[int], int]] = None,
        instrument_idx: Optional[Union[List[int], int]] = None,
    ) -> List[Note]:
        """beat_idx is the local index of the beat in the file."""
        first_idx = self.get_subbeats_in_bar(bar_idx)[0].beat_idx
        global_idx = beat_idx + first_idx
        all_notes = self.get_notes_in_bar(bar_idx, program, instrument_idx)
        return self._get_objs_in_beat(global_idx, all_notes) if all_notes is not None else []

    def get_subbeats_in_beat(self, beat_idx: int) -> List[Subdivision]:
        return self._get_objs_in_beat(beat_idx, self.subbeats)

    def get_subbeat_in_beat(self, subbeat_idx: int, beat_idx: int) -> Subdivision:
        all_subbeats = self._get_objs_in_beat(beat_idx, self.subbeats)
        # TODO: Error message if subbeat_idx > len(all_beats)
        return all_subbeats[subbeat_idx]

    def _get_objs_in_beat(self, beat_idx: int, objs):
        if beat_idx >= len(self.beats):
            raise ValueError(
                f"Not subbeat index {beat_idx} found in bars. The file has {len(self.beats)} beats."
            )
        return list(filter(lambda obj: obj.beat_idx == beat_idx, objs))

    # beats
    def get_notes_in_beats(
        self,
        beat_start: int,
        beat_end: int,
        program: Optional[Union[List[int], int]] = None,
        instrument_idx: Optional[Union[List[int], int]] = None,
    ) -> List[Note]:
        notes = self._filter_by_instruments(program, instrument_idx, self.notes)
        return self._get_objs_in_beats(
            beat_start, beat_end, notes
        ) if notes is not None else []

    def get_subbeats_in_beats(
        self,
        beat_start: int,
        beat_end: int,
    ) -> List[Subdivision]:
        return self._get_objs_in_beats(
            beat_start, beat_end, self.subbeats
        )

    def _get_objs_in_beats(
        self,
        beat_start: int,
        beat_end: int,
        objs,
    ):
        if beat_start > beat_end:
            raise ValueError("beat_start must be minor than beat_end.")
        return list(
            filter(
                lambda obj: obj.beat_idx >= beat_start and obj.beat_idx < beat_end, objs
            )
        )

    # Bar
    def get_notes_in_bar(
        self,
        bar_idx: int,
        program: Optional[Union[List[int], int]] = None,
        instrument_idx: Optional[Union[List[int], int]] = None,
    ) -> List[Note]:
        notes = self._filter_by_instruments(program, instrument_idx, self.notes)
        return self._get_objs_in_bar(bar_idx, notes) if notes is not None else []

    def get_beats_in_bar(
        self,
        bar_idx: int,
    ) -> List[Beat]:
        return self._get_objs_in_bar(bar_idx, self.beats)

    def get_beat_in_bar(self, beat_idx: int, bar_idx: int) -> Beat:
        all_beats = self._get_objs_in_bar(bar_idx, self.beats)
        # TODO: Error message if beat_idx > len(all_beats)
        return all_beats[beat_idx]

    def get_subbeats_in_bar(self, bar_idx: int) -> List[Subdivision]:
        return self._get_objs_in_bar(bar_idx, self.subbeats)

    def get_subbeat_in_bar(self, subbeat_idx: int, bar_idx: int) -> List[Subdivision]:
        all_subbeats = self._get_objs_in_bar(bar_idx, self.subbeats)
        # TODO: Error message if subbeat_idx > len(all_beats)
        return all_subbeats[subbeat_idx]

    def _get_objs_in_bar(
        self,
        bar_idx: int,
        objs: List[Note]
    ):
        if bar_idx >= len(self.bars):
            raise ValueError(
            f"Not bar index {bar_idx} found in bars. The file has {len(self.bars)} bars."
        )
        return list(filter(lambda obj: obj.bar_idx == bar_idx, objs))

    # Bars
    def get_notes_in_bars(
        self,
        bar_start: int,
        bar_end: int,
        program: Optional[Union[List[int], int]] = None,
        instrument_idx: Optional[Union[List[int], int]] = None,
    ) -> List[Note]:
        notes = self._filter_by_instruments(program, instrument_idx, self.notes)
        return self._get_objs_in_bars(bar_start, bar_end, notes) if notes is not None else []

    def get_beats_in_bars(self, bar_start: int, bar_end: int) -> List[Beat]:
        return self._get_objs_in_bars(bar_start, bar_end, self.beats)

    def get_subbeats_in_bars(
        self,
        bar_start: int,
        bar_end: int,
    ) -> List[Subdivision]:
        return self._get_objs_in_bars(bar_start, bar_end, self.subbeats)

    def _get_objs_in_bars(
        self,
        bar_start: int,
        bar_end: int,
        obj
    ):
        if bar_start > bar_end:
            raise ValueError("subbeat_start must be minor than subbeat_end.")
        return list(
            filter(
                lambda obj: obj.bar_idx >= bar_start and obj.bar_idx < bar_end, obj
            )
        )

    # Instruments
    def _filter_by_instruments(
        self,
        program: Optional[Union[List[int], int]],
        instrument_idx: Optional[Union[List[int], int]],
        objs,
    ):
        if program is not None:
            if isinstance(program, list):
                return self._filter_instruments(program, instrument_idx)
            elif isinstance(program, int):
                return self._filter_instrument(program, instrument_idx, objs)
        else:
            return objs

    def _filter_instruments(
        self,
        program: Optional[int],
        instrument_idx: Optional[List[int]],
    ):
        if instrument_idx is not None and len(program) != len(instrument_idx):
            raise ValueError("programs and instrument_idxs must have the same length.")
        diff_progs = list(set(program).difference(set(self.instruments_progs)))
        # if there's one or more programs not found, error
        if len(diff_progs) != 0:
            raise ValueError(
                f"Programs {diff_progs} not found. Instruments programs are {self.instruments_progs}."
            )
        if instrument_idx is None:
            instrument_idx = [None for _ in range(len(program))]
        objs = []
        for p, i in zip(program, instrument_idx):
            objs.extend(self.get_notes_in_bars(0, len(self.bars), p, i))
        objs.sort(key=lambda x: x.start_sec, reverse=False)
        return objs

    def _filter_instrument(
        self,
        program: Optional[int],
        instrument_idx: Optional[List[int]],
        objs,
    ):
        if program not in self.instruments_progs:
            raise ValueError(
                f"Not program {program} found in instruments. The file has the following programs: {self.instruments_progs}."
            )
        filtered = list(filter(lambda obj: obj.instrument_prog == program, objs))
        if instrument_idx is None:
            return filtered
        else:
            idxs = [note.instrument_idx for note in filtered]
            idxs = list(dict.fromkeys(idxs))
            if instrument_idx not in idxs:
                raise ValueError(
                    f"program {program} does not match instrument with index {instrument_idx}. "
                    f"Instrument indexes for program {program} are {idxs}."
                )
            return list(
                filter(lambda obj: obj.instrument_idx == instrument_idx, filtered)
            )

    @staticmethod
    def _get_last_note_end(pm_inst):
        # last note
        last_notes = [inst.notes[-1] for inst in pm_inst.instruments]
        last_notes.sort(key=lambda x: x.end, reverse=False)
        return last_notes[-1].end  # last note end in secs

    def writemidi(self, filename):
        midi = musa_to_prettymidi(self)
        midi.write(filename)

    def predict_key(self, method: str) -> str:
        """
        Predict the key with the key profiles algorithms.
        Note that signature fifths algorithm requires to initialize
        the Musa class with the argument `structure="bars"` instead
        of "instruments". The other algorithms work for both initializations.

        Parameters
        ----------

        method: str
            The algorithm we want to use to predict the key. The list of
            algorithms can be found here: :func:`~musicaiz.algorithms.KeyDetectionAlgorithms`.

        Raises
        ------

        ValueError

        ValueError

        Returns
        -------
        key: str
            The predicted key as a string separating tonic, alteration
            (if proceeds) and mode with "_".
        """
        if method not in KeyDetectionAlgorithms.all_values():
            raise ValueError("Not method found.")
        elif method in KeyDetectionAlgorithms.SIGNATURE_FIFTHS.value:
            # get notes in 2 1st bars (excluding drums)
            all_notes = self.get_notes_in_bars(0, 2)
            notes = [note for note in all_notes if not note.is_drum]
            key = key_detection(notes, method)
        elif (
            method in KeyDetectionAlgorithms.KRUMHANSL_KESSLER.value
            or KeyDetectionAlgorithms.TEMPERLEY.value
            or KeyDetectionAlgorithms.ALBRETCH_SHANAHAN.value
        ):
            notes = [note for note in self.notes if not note.is_drum]
            key = key_detection(notes, method)
        return key

    def _load_beats(self, last_note_end: float):
        # Populate bars considering time signature changes
        start_beat_ms = 0
        tempo_idx = 0
        for i, time_sig in enumerate(self.time_signature_changes):
            # latest note end will be the end of the time_sig_changes
            if i + 1 == len(self.time_signature_changes):
                ms_next_change = last_note_end * 1000
            else:
                # next time sig
                ms_next_change = self.time_signature_changes[i + 1]["ms"]

            while True:
                # we need to calculate the number of bars of time_sig[i] that we
                # have before the next change (sec_next_change)
                if tempo_idx + 1 >= len(self.tempo_changes):
                    break
                if start_beat_ms <= self.tempo_changes[tempo_idx + 1][
                    "ms"
                ] and tempo_idx < len(self.tempo_changes):
                    bpm = self.tempo_changes[tempo_idx]["tempo"]
                else:
                    tempo_idx += 1
                    bpm = self.tempo_changes[tempo_idx]["tempo"]

                # Get the duration in ms of one bar
                _, bar_ms = ms_per_bar(
                    time_sig["time_sig"].time_sig, bpm=bpm, resolution=self.resolution
                )
                beat_ms = bar_ms / time_sig["time_sig"].num
                beat_end = start_beat_ms + beat_ms

                if beat_end > ms_next_change:
                    beat_end = ms_next_change
                # If there's a tempo_change inside a bar, we'll
                # calculate the % of bar that is in each tempo to
                # calculate where the bar ends
                if tempo_idx + 1 < len(self.tempo_changes):
                    if beat_end > self.tempo_changes[tempo_idx + 1]["ms"]:
                        ms_in_prev_tempo = (
                            self.tempo_changes[tempo_idx + 1]["ms"] - start_beat_ms
                        )
                        perc = ms_in_prev_tempo / beat_ms
                        _, bar_ms = ms_per_bar(
                            time_sig["time_sig"].time_sig,
                            bpm=self.tempo_changes[tempo_idx + 1]["tempo"],
                            resolution=self.resolution,
                        )
                        beat_ms = bar_ms / time_sig["time_sig"].num
                        beat_end = start_beat_ms + ms_in_prev_tempo + beat_ms * perc

                beat = Beat(
                    time_sig=time_sig["time_sig"],
                    start=start_beat_ms / 1000,
                    end=beat_end / 1000,
                    bpm=self.tempo_changes[tempo_idx]["tempo"],
                    resolution=self.resolution,
                )
                self.beats.append(beat)
                start_beat_ms = beat.end_sec * 1000
                if start_beat_ms >= ms_next_change:
                    break

    # TODO: This is almost the same as _load_beats, refactor
    def _load_subdivisions(self, last_note_end: float):
        # Populate bars considering time signature changes
        start_subdiv_ms = 0
        tempo_idx = 0
        for i, time_sig in enumerate(self.time_signature_changes):
            # latest note end will be the end of the time_sig_changes
            if i + 1 == len(self.time_signature_changes):
                ms_next_change = last_note_end * 1000
            else:
                # next time sig
                ms_next_change = self.time_signature_changes[i + 1]["ms"]

            while True:
                # we need to calculate the number of bars of time_sig[i] that we
                # have before the next change (sec_next_change)
                if tempo_idx + 1 >= len(self.tempo_changes):
                    break
                if start_subdiv_ms <= self.tempo_changes[tempo_idx + 1][
                    "ms"
                ] and tempo_idx < len(self.tempo_changes):
                    bpm = self.tempo_changes[tempo_idx]["tempo"]
                else:
                    tempo_idx += 1
                    bpm = self.tempo_changes[tempo_idx]["tempo"]

                # Get the duration in ms of one bar
                _, bar_ms = ms_per_bar(
                    time_sig["time_sig"].time_sig, bpm=bpm, resolution=self.resolution
                )
                subdivs_in_bar = time_sig["time_sig"]._notes_per_bar(
                    self.subdivision_note.upper()
                )
                if subdivs_in_bar > 1:
                    ValueError("Subdivision note value must be lower than a bar.")
                subdiv_ms = bar_ms / subdivs_in_bar
                subdiv_end = start_subdiv_ms + subdiv_ms
                if subdiv_end > ms_next_change:
                    subdiv_end = ms_next_change
                # If there's a tempo_change inside a bar, we'll
                # calculate the % of bar that is in each tempo to
                # calculate where the bar ends
                if tempo_idx + 1 < len(self.tempo_changes):
                    if subdiv_end > self.tempo_changes[tempo_idx + 1]["ms"]:
                        ms_in_prev_tempo = (
                            self.tempo_changes[tempo_idx + 1]["ms"] - start_subdiv_ms
                        )
                        perc = ms_in_prev_tempo / subdiv_ms
                        _, bar_ms = ms_per_bar(
                            time_sig["time_sig"].time_sig,
                            bpm=self.tempo_changes[tempo_idx + 1]["tempo"],
                            resolution=self.resolution,
                        )
                        subdivs_in_bar = time_sig["time_sig"]._notes_per_bar(
                            self.subdivision_note.upper()
                        )
                        subdiv_ms = bar_ms / subdivs_in_bar
                        subdiv_end = start_subdiv_ms + ms_in_prev_tempo + subdiv_ms * perc

                subdiv = Subdivision(
                    time_sig=time_sig["time_sig"],
                    start=start_subdiv_ms / 1000,
                    end=subdiv_end / 1000,
                    bpm=self.tempo_changes[tempo_idx]["tempo"],
                    resolution=self.resolution,
                )
                self.subbeats.append(subdiv)
                self.subbeats[-1].global_idx = len(self.subbeats) - 1
                start_subdiv_ms = subdiv.end_sec * 1000
                if start_subdiv_ms >= ms_next_change:
                    break

    def _load_instruments_and_notes(self, pm_inst) -> List[Note]:
        """Populates `instruments` attribute mapping pretty_midi instruments
        to musicaiz instrument class."""
        # Load pretty midi instruments and notes
        notes = []
        for i, instrument in enumerate(pm_inst.instruments):
            self.instruments.append(
                Instrument(
                    program=instrument.program,
                    name=instrument.name,
                    is_drum=instrument.is_drum,
                    general_midi=self.general_midi,
                )
            )

            # convert pretty_midi Note objects to our Note objects
            t = 0
            for pm_note in instrument.notes:
                ms_next_change = self.tempo_changes[t + 1]["ms"]
                if pm_note.start * 1000 >= ms_next_change:
                    bpm = self.tempo_changes[t + 1]["tempo"]
                else:
                    bpm = self.tempo_changes[t]["tempo"]
                note = Note(
                    start=pm_note.start,
                    end=pm_note.end,
                    pitch=pm_note.pitch,
                    velocity=pm_note.velocity,
                    instrument_prog=instrument.program,
                    bpm=bpm,
                    resolution=self.resolution,
                    instrument_idx=i,
                    is_drum=instrument.is_drum,
                )
                notes.append(note)

        # sort notes by start time
        notes.sort(key=lambda x: x.start_sec, reverse=False)

        for note in notes:
            bar = list(
                filter(
                    lambda obj: obj[1].start_sec <= note.start_sec, enumerate(self.bars)
                )
            )
            note.bar_idx = bar[-1][0]
            beat = list(
                filter(
                    lambda obj: obj[1].start_sec <= note.start_sec, enumerate(self.beats)
                )
            )
            note.beat_idx = beat[-1][0]
            subbeat = list(
                filter(
                    lambda obj: obj[1].start_sec <= note.start_sec, enumerate(self.subbeats)
                )
            )
            note.subbeat_idx = subbeat[-1][0]
        self.notes = notes

    def _fill_bar_notes_attributes(self):
        for i, bar in enumerate(self.bars):
            notes = self.get_notes_in_bar(i)
            self.bars[i].note_density = len(notes)
            self.bars[i].harmonic_density = get_harmonic_density(notes)

    def _load_bars_and_group_beats_in_bars(self):
        bar_idx = 0
        beats = 0
        for i, beat in enumerate(self.beats):
            time_sig = beat.time_sig.time_sig
            beats_bar = beat.time_sig.num
            if i == 0:
                prev_beats_bar = beats_bar
                prev_time_sig = time_sig
                start_bar_sec = 0.0
            if beats_bar == prev_beats_bar and \
                beats + 1 <= beats_bar and \
                    time_sig == prev_time_sig:
                beat.bar_idx = bar_idx
                beats += 1
            else:
                end_bar_sec = beat.start_sec
                bar_idx += 1
                beats = 0
                beat.bar_idx = bar_idx
                beats += 1
                bar = Bar(
                    time_sig=self.beats[i - 1].time_sig,
                    start=start_bar_sec,
                    end=end_bar_sec,
                    bpm=self.beats[i - 1].bpm,
                    resolution=self.beats[i - 1].resolution
                )
                self.bars.append(bar)
                start_bar_sec = end_bar_sec
            beat.global_idx = i
            prev_beats_bar = beats_bar
            prev_time_sig = time_sig

        if self.beats[-1].end_ticks > self.bars[-1].end_ticks:
            # last bar
            beats_last_bar = [beat for beat in self.beats if beat.bar_idx == len(self.bars)]
            bar_sec = ms_per_bar(
                beats_last_bar[0].time_sig.time_sig,
                beats_last_bar[0].bpm,
                beats_last_bar[0].resolution
            )[1] / 1000
            bar = Bar(
                time_sig=beats_last_bar[0].time_sig,
                start=start_bar_sec,
                end=start_bar_sec + bar_sec,
                bpm=beat.bpm,
                resolution=beat.resolution
            )
            self.bars.append(bar)
            # Now add as musch beats as needed to complete the last bar
            if len(beats_last_bar) < bar.time_sig.num:
                beats = self.get_beats_in_bar(len(self.bars) - 1)
                for _ in range(bar.time_sig.num - len(beats_last_bar)):
                    dur = beats[0].end_sec - beats[0].start_sec
                    beat = Beat(
                        time_sig=beats[0].time_sig,
                        start=self.beats[-1].end_sec,
                        end=self.beats[-1].end_sec + dur,
                        bpm=beats[0].bpm,
                        resolution=self.beats[-1].resolution,
                    )
                    beat.global_idx = len(self.beats)
                    beat.bar_idx = len(self.bars) - 1
                    self.beats.append(beat)
