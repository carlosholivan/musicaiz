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


# Our modules
from musicaiz.structure import (
    Note,
    Instrument,
    Bar
)
from musicaiz.rhythm import (
    TimingConsts,
    get_subdivisions,
    ticks_per_bar,
    _bar_str_to_tuple,
    advanced_quantizer,
    get_ticks_from_subdivision,
    ms_per_tick,
    TimeSignature,
)
from musicaiz.features import harmony
from musicaiz.algorithms import (
    key_detection,
    KeyDetectionAlgorithms
)


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


STRUCTURES = ["bars", "instruments"]


class Musa:

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
                        self.time_sig = TimeSignature(str(msg.numerator) + "/" + str(msg.denominator))
                
                # initialize midi object with pretty_midi
                pm_inst = pm.PrettyMIDI(
                    midi_file=self.file,
                    resolution=self.resolution,
                    initial_tempo=self.bpm
                )
                # The MIDI file might not have the time signature not tempo information,
                # in that case, we initialize them as defalut (120bpm 4/4)
                if self.time_sig is None:
                    self.time_sig = TimeSignature(time_sig)

                # Divide notes into instrument and bars or just into instruments
                # depending on th evalue of the input argument `structure`
                if self.structure == "bars":
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
                            cut_notes=self.cut_notes
                        )
                    # TODO: All instr must have the same total_bars, we should get the track with more bars and
                    # append empty bars to the rest of the tracks
                    self.total_bars = len(self.instruments[0].bars)
                elif self.structure == "instruments":
                    self._load_instruments(pm_inst)
                    for instrument in self.instruments:
                        instrument.bars = None
                        self.notes.extend(instrument.notes)
                    self.total_bars = self.get_total_bars(self.notes)   
                else:
                    raise ValueError(f"Structure argument value {structure} is not valid.")

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
                    absolute_timing=self.absolute_timing
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

    def _load_instruments(self, pm_inst) -> List[Note]:
        """Populates `instruments` attribute mapping pretty_midi instruments
        to musicaiz instrument class."""
        # Load pretty midi instrument
        for i, instrument in enumerate(pm_inst.instruments):
            self.instruments.append(
                Instrument(
                    program=instrument.program,
                    name=instrument.name,
                    is_drum=instrument.is_drum,
                    general_midi=False,
                )
            )
            for note in instrument.notes:
                # Initialize the note with musicaiz `Note` object
                self.instruments[i].notes.append(
                    Note(
                        pitch=note.pitch,
                        start=note.start,
                        end=note.end,
                        velocity=note.velocity,
                        bpm=self.bpm,
                        resolution=self.resolution,
                    )
                )

    def _load_bars(self):
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

    def _load_bars_notes(
        self,
        instrument: Instrument,
        cut_notes: bool = False,
        absolute_timing: bool = True
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
                bar.start_sec = bar.start_ticks * ms_per_tick(self.bpm, self.resolution) / 1000
            else:
                bar.start_ticks, bar.start_sec = 0, 0.0
                bar.end_ticks = bar.start_ticks + bar_ticks
            bar.end_sec = bar.end_ticks * ms_per_tick(self.bpm, self.resolution) / 1000

            for i, note in enumerate(instrument.notes):
                # TODO: If note ends after the next bar start? Fix this, like this we'll loose it
                if note.start_ticks >= start_bar_ticks and note.end_ticks <= next_start_bar_ticks:
                    bar.notes.append(note)
                # note starts in current bar but ends in the next (or nexts bars) -> cut note
                elif start_bar_ticks <= note.start_ticks <= next_start_bar_ticks and note.end_ticks >= next_start_bar_ticks:
                    if cut_notes:
                        # cut note by creating a new note that starts when the next bar starts
                        note_next_bar = Note(
                            start=next_start_bar_ticks, end=note.end_ticks,
                            pitch=note.pitch, velocity=note.velocity, ligated=True
                        )
                        notes_next_bar.append(note_next_bar)
                        # cut note by assigning end note to the current end bar
                        note.end_ticks = next_start_bar_ticks
                        note.end_secs = next_start_bar_ticks * ms_per_tick(self.bpm, self.resolution) / 1000
                        note.ligated = True
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
        elif method in KeyDetectionAlgorithms.KRUMHANSL_KESSLER.value or \
            KeyDetectionAlgorithms.TEMPERLEY.value or \
                KeyDetectionAlgorithms.ALBRETCH_SHANAHAN.value:
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
            'set_tempo': lambda e: (1 * 256 * 256),
            'time_signature': lambda e: (2 * 256 * 256),
            'key_signature': lambda e: (3 * 256 * 256),
            'lyrics': lambda e: (4 * 256 * 256),
            'text_events' :lambda e: (5 * 256 * 256),
            'program_change': lambda e: (6 * 256 * 256),
            'pitchwheel': lambda e: ((7 * 256 * 256) + e.pitch),
            'control_change': lambda e: (
                (8 * 256 * 256) + (e.control * 256) + e.value),
            'note_off': lambda e: ((9 * 256 * 256) + (e.note * 256)),
            'note_on': lambda e: (
                (10 * 256 * 256) + (e.note * 256) + e.velocity),
            'end_of_track': lambda e: (11 * 256 * 256)
        }
        # If the events have the same tick, and both events have types
        # which appear in the secondary_sort dictionary, use the dictionary
        # to determine their ordering.
        if (event1.time == event2.time and
                event1.type in secondary_sort and
                event2.type in secondary_sort):
            return (secondary_sort[event1.type](event1) -
                    secondary_sort[event2.type](event2))
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
                "set_tempo", time=0,
                # Convert from microseconds per quarter note to BPM
                tempo=self.bpm)
        )
        #Write key TODO
        #timing_track.append(
            #mido.MetaMessage("key_signature", time=self.time_to_tick(ks.time),
            #key=key_number_to_mido_key_name[ks.key_number]))

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
                track.append(mido.MetaMessage(
                    'track_name', time=0, name=instrument.name))
            # If it's a drum event, we need to set channel to 9
            if instrument.is_drum:
                channel = 9
            # Otherwise, choose a channel from the possible channel list
            else:
                channel = 8  # channels[n % len(channels)]
            # Set the program number
            track.append(mido.Message(
                'program_change', time=0, program=instrument.program,
                channel=channel))
            # Add all note events
            ligated_notes = []
            for idx, note in enumerate(instrument.notes):
                if note.ligated:
                    ligated_notes.append(note)
                    for next_note in instrument.notes[idx+1:]:
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
                track.append(mido.Message(
                    'note_on', time=note.start_ticks,
                    channel=channel, note=note.pitch, velocity=note.velocity))
                # Also need a note-off event (note on with velocity 0)
                track.append(mido.Message(
                    'note_on', time=note.end_ticks,
                    channel=channel, note=note.pitch, velocity=0))

            # Sort all the events using the event_compare comparator.
            track = sorted(track, key=functools.cmp_to_key(self._event_compare))

            # If there's a note off event and a note on event with the same
            # tick and pitch, put the note off event first
            for n, (event1, event2) in enumerate(zip(track[:-1], track[1:])):
                if (event1.time == event2.time and
                        event1.type == 'note_on' and
                        event2.type == 'note_on' and
                        event1.note == event2.note and
                        event1.velocity != 0 and
                        event2.velocity == 0):
                    track[n] = event2
                    track[n + 1] = event1
            # Finally, add in an end of track event
            track.append(mido.MetaMessage(
                'end_of_track', time=track[-1].time + 1))
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
        if len(self.instruments) == 0 or all(len(i.notes) == 0 for i in self.instruments):
            return np.array([])
        # Get synthesized waveform for each instrument
        waveforms = [i.fluidsynth(fs=fs,
                                  sf2_path=sf2_path) for i in self.instruments]
        # Allocate output waveform, with #sample = max length of all waveforms
        synthesized = np.zeros(np.max([w.shape[0] for w in waveforms]))
        # Sum all waveforms in
        for waveform in waveforms:
            synthesized[:waveform.shape[0]] += waveform
        # Normalize
        synthesized /= np.abs(synthesized).max()
        return synthesized