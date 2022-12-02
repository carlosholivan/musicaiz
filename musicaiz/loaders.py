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
from typing import TextIO, Union, List, Optional
import pretty_midi as pm
from pathlib import Path
from enum import Enum
import mido
import functools
import numpy as np
from traitlets import Callable


# Our modules
from musicaiz.structure import Note, Instrument, Bar
from musicaiz.errors import BarIdxErrorMessage
from musicaiz.rhythm import (
    TimingConsts,
    get_subdivisions,
    ticks_per_bar,
    _bar_str_to_tuple,
    advanced_quantizer,
    get_ticks_from_subdivision,
    ms_per_tick,
    ms_per_bar,
    TimeSignature,
    Beat,
    Subdivision,
)
from musicaiz.converters import musa_to_prettymidi
from musicaiz.features import get_harmonic_density
from musicaiz.algorithms import key_detection, KeyDetectionAlgorithms
from tests.unit.musicaiz import converters


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


class MusaII:

    """Musanalisys main object. This object loads a file and maps it to the
    musicaiz' objects defined in the submodules `harmony` and `structure`.

    Attributes
    ----------

    file: Union[str, TextIO]
        The input file. It can be a MIDI file, a MusicXML file (TODO) or and ABC file (TODO).

    structure: str
        Organices the attributes at different structure levels which are bar,
        instrument or piece level.
        Defaults to "piece".

    quantize: bool
        Default is True. Quantizes the notes at bar or instrument level with the
        `rhythm.advanced_quantizer` method that uses a strength of 100%.


    tonality: Optional[str]
        Initializes the MIDI file and adds the tonality attribute. Knowing the tonality
        in advance means that notes are initialized by knowing their name
        (ex.: pitch 24 can be C or B#) so this reduces the complexity of the chord and
        key prediction algorithms.

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
        default is True. This allows to initialize note time arguments in absolute (True) or
        relative time units (False). Relative units means that each bar will start at 0 seconds
        and ticks, so the note timing attributes will be relative to the bar start equals to 0.

    Raises
    ------

    ValueError: [description]
    """

    __slots__ = [
        "file",
        "structure",
        "tonality",
        "time_sig",
        "bpm",
        "resolution",
        "data",
        "instruments",
        "bars",
        "is_quantized",
        "notes",
        "total_bars",
        "absolute_timing",
        "cut_notes",
    ]

    def __init__(
        self,
        file: Optional[Union[str, TextIO, Path]] = None,
        structure: str = "instruments",
        quantize: bool = False,
        quantize_note: Optional[str] = "sixteenth",
        cut_notes: bool = False,
        tonality: Optional[str] = None,
        time_sig: str = TimingConsts.DEFAULT_TIME_SIGNATURE.value,
        bpm: int = TimingConsts.DEFAULT_BPM.value,
        resolution: int = TimingConsts.RESOLUTION.value,
        absolute_timing: bool = True,
    ):

        self.file = None
        self.data = None
        self.notes = []
        self.tonality = tonality
        self.time_sig = None
        self.bpm = bpm
        self.resolution = resolution
        self.instruments = []
        self.bars = []
        self.structure = structure
        self.total_bars = None
        self.absolute_timing = absolute_timing
        self.is_quantized = quantize
        self.cut_notes = cut_notes

        if quantize_note != "eight" and quantize_note != "sixteenth":
            raise ValueError("quantize_note must be sixteenth or eight")

        # TODO: What if time_sig, tonality and bpm change inside the piece?

        # File provided
        if file is not None:
            if isinstance(file, Path):
                file = str(file)
            if self.is_valid(file):
                self.file = file
            else:
                raise ValueError("Input file extension is not valid.")

            if self.is_midi(file):
                # Read bpm from MIDI file with mido
                m = mido.MidiFile(self.file)
                for msg in m:
                    if msg.type == "set_tempo":
                        self.bpm = int(mido.tempo2bpm(msg.tempo))
                    elif msg.type == "time_signature":
                        self.time_sig = TimeSignature(
                            str(msg.numerator) + "/" + str(msg.denominator)
                        )

                # initialize midi object with pretty_midi
                pm_inst = pm.PrettyMIDI(
                    midi_file=self.file,
                    resolution=self.resolution,
                    initial_tempo=self.bpm,
                )
                # The MIDI file might not have the time signature not tempo information,
                # in that case, we initialize them as defalut (120bpm 4/4)
                if self.time_sig is None:
                    self.time_sig = TimeSignature(time_sig)

                # Divide notes into instrument and bars or just into instruments
                # depending on th evalue of the input argument `structure`
                if self.structure == "instrument_bars":
                    # Map instruments to load them as musicaiz instrument class
                    self._load_instruments(pm_inst)
                    self.notes = []
                    self._load_inst_bars()
                    for instrument in self.instruments:
                        self._load_bars_notes(
                            instrument,
                            absolute_timing=self.absolute_timing,
                            cut_notes=self.cut_notes,
                        )
                    # TODO: All instr must have the same total_bars, we should get the track with more bars and
                    # append empty bars to the rest of the tracks
                    self.total_bars = len(self.instruments[0].bars)
                elif self.structure == "bars":
                    # Map instruments to load them as musicaiz instrument class
                    self._load_instruments(pm_inst)
                    self.notes = []
                    # Concatenate all the notes of different instruments
                    # this is for getting the latest note of the piece
                    # and get the total number of bars of the piece
                    for instrument in self.instruments:
                        self.notes.extend(instrument.notes)
                    self._load_bars()
                    for instrument in self.instruments:
                        self._load_bars_notes(
                            instrument,
                            absolute_timing=self.absolute_timing,
                            cut_notes=self.cut_notes,
                        )
                    # TODO: All instr must have the same total_bars, we should get the track with more bars and
                    # append empty bars to the rest of the tracks
                    self.total_bars = len(self.instruments[0].bars)
                elif self.structure == "notes":
                    # Concatenate all the notes of different instruments
                    # this is for getting the latest note of the piece
                    # and get the total number of bars of the piece
                    for instrument in pm_inst.instruments:
                        self.notes.extend(instrument.notes)
                    self.instruments = []
                    self.bars = []
                elif self.structure == "instruments":
                    self._load_instruments(pm_inst)
                    for instrument in self.instruments:
                        instrument.bars = None
                        self.notes.extend(instrument.notes)
                    self.total_bars = self.get_total_bars(self.notes)
                else:
                    raise ValueError(
                        f"Structure argument value {structure} is not valid."
                    )

            elif self.is_musicxml(file):
                # initialize musicxml object with ??
                # TODO: implement musicxml parser
                self.data = None

            # Now quantize if is_quantized
            if quantize:
                grid = get_subdivisions(
                    total_bars=self.total_bars,
                    subdivision=quantize_note,
                    time_sig=self.time_sig.time_sig,
                    bpm=self.bpm,
                    resolution=self.resolution,
                    absolute_timing=self.absolute_timing,
                )
                v_grid = get_ticks_from_subdivision(grid)
                for instrument in self.instruments:
                    if self.structure == "bars":
                        for bar in instrument.bars:
                            advanced_quantizer(bar.notes, v_grid)
                    advanced_quantizer(instrument.notes, v_grid)

        # sort the notes in all the midi file
        self.notes.sort(key=lambda x: x.start_ticks, reverse=False)

    @classmethod
    def is_valid(cls, file: Union[str, TextIO]):
        extension = cls.get_file_extension(file)
        return True if extension in ValidFiles.all_extensions() else False

    # TODO: How to split if arg is a filepointer?
    @staticmethod
    def get_file_extension(file: Union[str, TextIO]):
        return Path(file).suffix

    @classmethod
    def is_midi(cls, file: Union[str, TextIO]):
        extension = cls.get_file_extension(file)
        return True if extension in ValidFiles.MIDI.value else False

    @classmethod
    def is_musicxml(cls, file: Union[str, TextIO]):
        extension = cls.get_file_extension(file)
        return True if extension in ValidFiles.MUSIC_XML.value else False

    def _load_inst_bars(self):
        """Load the bars for an instrument."""
        total_bars = self.get_total_bars(self.notes)
        for instrument in self.instruments:
            for _ in range(total_bars):
                instrument.bars.append(
                    Bar(
                        time_sig=self.time_sig.time_sig,
                        bpm=self.bpm,
                    )
                )

    def _load_bars(self):
        """Load the bars for an instrument."""
        total_bars = self.get_total_bars(self.notes)
        for instrument in pm_inst.instruments:
            for _ in range(total_bars):
                instrument.bars.append(
                    Bar(
                        time_sig=self.time_sig.time_sig,
                        bpm=self.bpm,
                    )
                )

    def _load_bars_notes(
        self,
        instrument: Instrument,
        cut_notes: bool = False,
        absolute_timing: bool = True,
    ):
        start_bar_ticks = 0
        _, bar_ticks = ticks_per_bar(self.time_sig.time_sig, self.resolution)
        notes_next_bar = []
        for bar_idx, bar in enumerate(instrument.bars):
            for n in notes_next_bar:
                bar.notes.append(n)
            notes_next_bar = []
            next_start_bar_ticks = start_bar_ticks + bar_ticks
            # bar obj attributes
            if self.absolute_timing:
                bar.start_ticks = start_bar_ticks
                bar.end_ticks = next_start_bar_ticks
                bar.start_sec = (
                    bar.start_ticks * ms_per_tick(self.bpm, self.resolution) / 1000
                )
            else:
                bar.start_ticks, bar.start_sec = 0, 0.0
                bar.end_ticks = bar.start_ticks + bar_ticks
            bar.end_sec = bar.end_ticks * ms_per_tick(self.bpm, self.resolution) / 1000

            for i, note in enumerate(instrument.notes):
                # TODO: If note ends after the next bar start? Fix this, like this we'll loose it
                if (
                    note.start_ticks >= start_bar_ticks
                    and note.end_ticks <= next_start_bar_ticks
                ):
                    bar.notes.append(note)
                # note starts in current bar but ends in the next (or nexts bars) -> cut note
                elif (
                    start_bar_ticks <= note.start_ticks <= next_start_bar_ticks
                    and note.end_ticks >= next_start_bar_ticks
                ):
                    if cut_notes:
                        # cut note by creating a new note that starts when the next bar starts
                        note_next_bar = Note(
                            start=next_start_bar_ticks,
                            end=note.end_ticks,
                            pitch=note.pitch,
                            velocity=note.velocity,
                            ligated=True,
                        )
                        notes_next_bar.append(note_next_bar)
                        # cut note by assigning end note to the current end bar
                        note.end_ticks = next_start_bar_ticks
                        note.end_secs = (
                            next_start_bar_ticks
                            * ms_per_tick(self.bpm, self.resolution)
                            / 1000
                        )
                        note.ligated = True
                        note.instrument_prog = instrument.program
                    bar.notes.append(note)
                elif note.start_ticks > next_start_bar_ticks:
                    break

            # sort notes in the bar by their onset
            bar.notes.sort(key=lambda x: x.start_ticks, reverse=False)

            # update bar attributes now that we know its notes
            bar.note_density = len(bar.notes)
            bar.harmonic_density = harmony.get_harmonic_density(bar.notes)
            # if absolute_timing is False, we'll write the note time attributes relative
            # to their corresponding bar
            if not absolute_timing:
                bar.relative_notes_timing(bar_start=start_bar_ticks)
            start_bar_ticks = next_start_bar_ticks

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
            if self.structure == "bars":
                notes = []
                for inst in self.instruments:
                    for b, bar in enumerate(inst.bars):
                        # Signature fifths only takes the 2 1st bars of the piece
                        if b == 2:
                            break
                        notes.extend(bar.notes)
                notes.sort(key=lambda x: x.start_ticks, reverse=False)
                key = key_detection(notes, method)
            elif self.structure == "instruments":
                raise ValueError("Initialize the Musa with `structure=bars`")
        elif (
            method in KeyDetectionAlgorithms.KRUMHANSL_KESSLER.value
            or KeyDetectionAlgorithms.TEMPERLEY.value
            or KeyDetectionAlgorithms.ALBRETCH_SHANAHAN.value
        ):
            key = key_detection(self.notes, method)
        return key

    @staticmethod
    def group_instrument_bar_notes(musa_object: Musa) -> List[Bar]:
        """Instead of having the structure Instrument -> Bar -> Note, this
        method group groups the infrmation as: Bar -> Note."""
        bars = []
        for inst_idx, instrument in enumerate(musa_object.instruments):
            for bar_idx, bar in enumerate(instrument.bars):
                if inst_idx == 0:
                    bars.append([])
                bars[bar_idx].extend(bar.notes)
        return bars

    # TODO: Move this to `utils.py`?
    @staticmethod
    def _last_note(note_seq: List[Note]) -> Union[Note, None]:
        """Get the last note of a sequence."""
        # If there's only 1 note in the sequence, that'll be the
        # latest note, so we initialize t to -1 so at least
        # 1 note in the note seq. raises the if condition
        if len(note_seq) == 0:
            return None
        t = -1
        for n in note_seq:
            if n.end_ticks > t:
                last_note = n
                t = n.end_ticks
        return last_note

    def get_total_bars(self, note_seq: List[Note]):
        """
        Calculates the number of bars of a sequence.
        """
        last_note = self._last_note(note_seq)
        # TODO: Detect number of bars if time signatura changes
        subdivisions = get_subdivisions(
            total_bars=500,  # initialize with a big number of bars
            subdivision=self.time_sig.beat_type.upper(),
            time_sig=self.time_sig.time_sig,
            bpm=self.bpm,
            resolution=self.resolution,
        )
        # We calculate the number of bars of the piece (supposing same time_sig)
        for s in subdivisions:
            if last_note.end_ticks > s["ticks"]:
                total_bars = s["bar"]
        return total_bars

    @staticmethod
    def _event_compare(event1, event2):
        """Compares two events for sorting.
        Events are sorted by tick time ascending. Events with the same tick
        time ares sorted by event type. Some events are sorted by
        additional values. For example, Note On events are sorted by pitch
        then velocity, ensuring that a Note Off (Note On with velocity 0)
        will never follow a Note On with the same pitch.
        Parameters
        ----------
        event1, event2 : mido.Message
            Two events to be compared.
        """
        # Construct a dictionary which will map event names to numeric
        # values which produce the correct sorting.  Each dictionary value
        # is a function which accepts an event and returns a score.
        # The spacing for these scores is 256, which is larger than the
        # largest value a MIDI value can take.
        secondary_sort = {
            "set_tempo": lambda e: (1 * 256 * 256),
            "time_signature": lambda e: (2 * 256 * 256),
            "key_signature": lambda e: (3 * 256 * 256),
            "lyrics": lambda e: (4 * 256 * 256),
            "text_events": lambda e: (5 * 256 * 256),
            "program_change": lambda e: (6 * 256 * 256),
            "pitchwheel": lambda e: ((7 * 256 * 256) + e.pitch),
            "control_change": lambda e: ((8 * 256 * 256) + (e.control * 256) + e.value),
            "note_off": lambda e: ((9 * 256 * 256) + (e.note * 256)),
            "note_on": lambda e: ((10 * 256 * 256) + (e.note * 256) + e.velocity),
            "end_of_track": lambda e: (11 * 256 * 256),
        }
        # If the events have the same tick, and both events have types
        # which appear in the secondary_sort dictionary, use the dictionary
        # to determine their ordering.
        if (
            event1.time == event2.time
            and event1.type in secondary_sort
            and event2.type in secondary_sort
        ):
            return secondary_sort[event1.type](event1) - secondary_sort[event2.type](
                event2
            )
        # Otherwise, just return the difference of their ticks.
        return event1.time - event2.time

    def write_midi(self, filename: str):
        """Writes a Musa object to a MIDI file.
        This is adapted from `pretty_midi` library."""
        pass
        # TODO: Support tempo, time sig. and key changes
        # Initialize output MIDI object
        mid = mido.MidiFile(ticks_per_beat=self.resolution)

        # Create track 0 with timing information
        timing_track = mido.MidiTrack()

        # Write time sig.
        num, den = _bar_str_to_tuple(self.time_sig.time_sig)
        timing_track.append(
            mido.MetaMessage("time_signature", time=0, numerator=num, denominator=den)
        )
        # Write BPM
        timing_track.append(
            mido.MetaMessage(
                "set_tempo",
                time=0,
                # Convert from microseconds per quarter note to BPM
                tempo=self.bpm,
            )
        )
        # Write key TODO
        # timing_track.append(
        # mido.MetaMessage("key_signature", time=self.time_to_tick(ks.time),
        # key=key_number_to_mido_key_name[ks.key_number]))

        for n, instrument in enumerate(self.instruments):
            # Perharps notes are grouped in bars, concatenate them
            if len(instrument.notes) == 0:
                # TODO: check if note have absolute durations, relative ones won't work
                for bar in instrument.bars:
                    instrument.notes.extend(bar.notes)
            # Initialize track for this instrument
            track = mido.MidiTrack()
            # Add track name event if instrument has a name
            if instrument.name:
                track.append(
                    mido.MetaMessage("track_name", time=0, name=instrument.name)
                )
            # If it's a drum event, we need to set channel to 9
            if instrument.is_drum:
                channel = 9
            # Otherwise, choose a channel from the possible channel list
            else:
                channel = 8  # channels[n % len(channels)]
            # Set the program number
            track.append(
                mido.Message(
                    "program_change",
                    time=0,
                    program=instrument.program,
                    channel=channel,
                )
            )
            # Add all note events
            ligated_notes = []
            for idx, note in enumerate(instrument.notes):
                if note.ligated:
                    ligated_notes.append(note)
                    for next_note in instrument.notes[idx + 1 :]:
                        if not next_note.ligated:
                            continue
                        ligated_notes.append(next_note)
                    # Concat all ligated notes into one note by rewriting the 1st ligated note args
                    note.start_ticks = ligated_notes[0].start_ticks
                    note.start_sec = ligated_notes[0].start_sec
                    note.end_ticks = ligated_notes[-1].end_ticks
                    note.end_sec = ligated_notes[-1].end_sec
                ligated_notes = []
                # Construct the note-on event
                track.append(
                    mido.Message(
                        "note_on",
                        time=note.start_ticks,
                        channel=channel,
                        note=note.pitch,
                        velocity=note.velocity,
                    )
                )
                # Also need a note-off event (note on with velocity 0)
                track.append(
                    mido.Message(
                        "note_on",
                        time=note.end_ticks,
                        channel=channel,
                        note=note.pitch,
                        velocity=0,
                    )
                )

            # Sort all the events using the event_compare comparator.
            track = sorted(track, key=functools.cmp_to_key(self._event_compare))

            # If there's a note off event and a note on event with the same
            # tick and pitch, put the note off event first
            for n, (event1, event2) in enumerate(zip(track[:-1], track[1:])):
                if (
                    event1.time == event2.time
                    and event1.type == "note_on"
                    and event2.type == "note_on"
                    and event1.note == event2.note
                    and event1.velocity != 0
                    and event2.velocity == 0
                ):
                    track[n] = event2
                    track[n + 1] = event1
            # Finally, add in an end of track event
            track.append(mido.MetaMessage("end_of_track", time=track[-1].time + 1))
            # Add to the list of output tracks
            mid.tracks.append(track)
        # Turn ticks to relative time from absolute
        for track in mid.tracks:
            tick = 0
            for event in track:
                event.time -= tick
                tick += event.time
        mid.save(filename + ".mid")

    def fluidsynth(self, fs=44100, sf2_path=None):
        """Synthesize using fluidsynth.
        Parameters
        ----------
        fs : int
            Sampling rate to synthesize at.
        sf2_path : str
            Path to a .sf2 file.
            Default ``None``, which uses the TimGM6mb.sf2 file included with
            ``pretty_midi``.
        Returns
        -------
        synthesized : np.ndarray
            Waveform of the MIDI data, synthesized at ``fs``.
        """
        # If there are no instruments, or all instruments have no notes, return
        # an empty array
        if len(self.instruments) == 0 or all(
            len(i.notes) == 0 for i in self.instruments
        ):
            return np.array([])
        # Get synthesized waveform for each instrument
        waveforms = [i.fluidsynth(fs=fs, sf2_path=sf2_path) for i in self.instruments]
        # Allocate output waveform, with #sample = max length of all waveforms
        synthesized = np.zeros(np.max([w.shape[0] for w in waveforms]))
        # Sum all waveforms in
        for waveform in waveforms:
            synthesized[: waveform.shape[0]] += waveform
        # Normalize
        synthesized /= np.abs(synthesized).max()
        return synthesized



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
        "quantize_note",
        "general_midi",
        "subdivision_note",
        "subbeats",
        "beats",
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
        quantize_note: Optional[str] = "sixteenth",
        cut_notes: bool = False,
        tonality: Optional[str] = None,
        resolution: Optional[int] = None,
        absolute_timing: bool = True,
        general_midi: bool = False,
        subdivision_note: str = "sixteenth"
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
        # TODO: key signature changes, 
        # TODO: write_midi -> with pretty_midi
        # TODO: synthesize -> with pretty_midi

        self.instruments = []
        self.notes = []
        self.total_bars = 0
        self.general_midi = general_midi
        self.absolute_timing = absolute_timing
        self.is_quantized = quantize
        self.quantize_note = quantize_note
        self.subdivision_note = subdivision_note
        self.subbeats = []
        self.beats = []
        self.cut_notes = cut_notes
        self.time_signature_changes = []
        self.bars = []
        self.tempo_changes = []

        # TODO unify quantize_note as subdivision_note?

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


            # group subdivisions in beats


            # group subdivisions in bars

    def json(self):
        return {key : getattr(self, key, None) for key in self.__slots__}

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

        # For whatever reason, get_tempo_changes() returns a tuple
        # of arrays with 2 elements per array, the 1st is the time and
        # the 2nd is the tempo. In other cases, only 2 arrays are returned
        # each one with one element, the 1st array is the time and the 2nd, the tempo
        tempo_changes = pm_inst.get_tempo_changes()
        if tempo_changes[0].size == 2:
            for tempo_changes in tempo_changes:
                if tempo_changes.size == 2:
                    self.tempo_changes.append(
                        {"tempo": tempo_changes[1], "ms": tempo_changes[0] * 1000}
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
        # (for easier bars laoding)
        self.tempo_changes.append(
            {"tempo": self.tempo_changes[-1]["tempo"], "ms": last_note_end * 1000}
        )

        # Load Bars
        #self._load_bars(last_note_end)

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
        """
        # if quantize
        if quantize:
            grid = get_subdivisions(
                total_bars=self.total_bars,
                subdivision=quantize_note,
                time_sig=self.time_sig.time_sig,
                bpm=self.bpm,
                resolution=self.resolution,
                absolute_timing=self.absolute_timing,
            )
            v_grid = get_ticks_from_subdivision(grid)
            advanced_quantizer(self.notes, v_grid)
        """

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
            # TODO: Exclude drums
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
                for _ in range(bar.time_sig.num - len(beats_last_bar)):
                    dur = self.beats[-1].end_sec - self.beats[-1].start_sec
                    beat = Beat(
                        time_sig=self.beats[-1].time_sig,
                        start=self.beats[-1].end_sec,
                        end=self.beats[-1].end_sec + dur,
                        bpm=self.beats[-1].bpm,
                        resolution=self.beats[-1].resolution,
                    )
                    beat.global_idx = len(self.beats)
                    beat.bar_idx = len(self.bars) - 1
                    self.beats.append(beat)
