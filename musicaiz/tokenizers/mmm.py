
from typing import Optional, List, Dict, Union, TextIO
from pathlib import Path
import logging
from dataclasses import dataclass

from musicaiz.loaders import Musa
from musicaiz.structure import Note, Instrument, Bar
from musicaiz.tokenizers import EncodeBase, TokenizerArguments
from musicaiz.rhythm import (
    TimingConsts,
    ticks_per_bar,
    get_symbolic_duration,
    NoteLengths,
    TimeSignature,
)


# time units available to tokenize
VALID_TIME_UNITS = ["SIXTEENTH", "THIRTY_SECOND", "SIXTY_FOUR", "HUNDRED_TWENTY_EIGHT"]


logger = logging.getLogger("mmm-tokenizer")
logging.basicConfig(level=logging.INFO)


@dataclass
class MMMTokenizerArguments(TokenizerArguments):
    """
    prev_tokens: str
        if we want to add tokens after the `PIECE_START` token and before
        the 1st TRACK_START token (for conditioning...).

    windowing: bool
        if True, the method tokenizes each file by applying bars windowing.

    time_unit: str
        the note length in `VALID_TIME_UNITS` that one `TIME_DELTA` unit will be equal to.
        This allows to tokenize in a wide variety of note lengths for diverse purposes.
        Be careful when choosing this value because if there are notes which duration is
        lower than the chosen time_unit value, they won't be tokenized.

    num_programs: List[int]
        the number of programs to tokenize. If None, the method tokenizes all the tracks.

    shuffle_tracks: bool
        shuffles the order of tracks in each window (PIECE).

    track_density: bool
        if True a token DENSITY is added at the beggining of each track.

    window_size: int
        the number of bars per track to tokenize.

    hop_length: int
        the number of bars to slice when tokenizing.
        If a MIDI file contains 5 bars and the window size is 4 and the hop length is 1,
        it'll be splitted in 2 PIECE tokens, one from bar 1 to 4 and the other on from
        bar 2 to 5 (somehow like audio FFT).

    time_sig: bool
        if we want to include the time signature in the samples. Note that the time signature
        will be added to the piece-level, that is, before the first track starts.

    velocity: bool
        if we want to add the velocity token. Velocities ranges between 1 and 128 (ints).

    quantize: bool
        if we want to quantize the symbolic music data for tokenizing.
    """

    prev_tokens: str = ""
    windowing: bool = True
    time_unit: str = "THIRTY_SECOND"
    num_programs: Optional[List[int]] = None
    shuffle_tracks: bool = True
    track_density: bool = False
    window_size: int = 4
    hop_length: int = 1
    time_sig: bool = False
    velocity: bool = False
    quantize: bool = False
    tempo: bool = True


class MMMTokenizer(EncodeBase):
    """
    This class presents methods to compute the Multi-Track Music Machine Encoding.

    Attributes
    ----------
    file: Optional[Union[str, TextIO, Path]] = None
    """

    def __init__(
        self,
        file: Optional[Union[str, TextIO, Path]] = None,
        args: MMMTokenizerArguments = None
    ):

        if args is None:
            raise ValueError("No `MMMTokenizerArguments` passed.")
        self.args = args

        # Convert file into a Musa object to be processed
        if file is not None:
            self.midi_object = Musa(
                file=file,
                absolute_timing=False,
                quantize=self.args.quantize,
                cut_notes=False
            )
        else:
            self.midi_object = Musa(file=None)

    def tokenize_file(
        self,
    ) -> str:
        """
        This method tokenizes a Musa (MIDI) object.

        Returns
        -------

        all_tokens: List[str]
            the list of tokens corresponding to all the windows.
        """
        # Do not tokenize the tracks that are not in num_programs
        # but if num_programs is None then tokenize all instruments

        tokenized_instruments = []
        if self.args.num_programs is not None:
            for inst in self.midi_object:
                if inst.program in self.args.num_programs:
                    tokenized_instruments.append(inst)
        else:
            tokenized_instruments = self.midi_object.instruments

        if not self.args.windowing:
            if self.args.time_sig:
                time_sig = self.midi_object.time_signature_changes[0]['time_sig'].time_sig
                time_sig_tok = f"TIME_SIG={time_sig} "
            else:
                time_sig_tok = ""
            if self.args.tempo:
                tempo_tok = f"TEMPO={self.midi_object.tempo_changes[0]['tempo']} "
            else:
                tempo_tok = ""
            tokens = self.tokenize_tracks(
                instruments=tokenized_instruments,
                bar_start=0,
                tokens="PIECE_START " + self.args.prev_tokens + time_sig_tok + tempo_tok,
            )
            tokens += "\n"
        else:
            # Now tokenize and create a PIECE for each window
            # that is defined in terms of bars
            # loop in bars
            tokens = ""
            for i in range(0, self.midi_object.total_bars, self.args.hop_length):
                if i + self.args.window_size == self.midi_object.total_bars:
                    break
                if self.args.time_sig:
                    time_sig = self.midi_object.time_signature_changes[0]['time_sig'].time_sig
                    time_sig_tok = f"TIME_SIG={time_sig} "
                else:
                    time_sig_tok = ""
                tokens += self.tokenize_tracks(
                    tokenized_instruments,
                    bar_start=i,
                    bar_end=i + self.args.window_size,
                    tokens="PIECE_START " + self.args.prev_tokens + time_sig_tok,
                )
                tokens += "\n"
        return tokens

    def tokenize_tracks(
        self,
        instruments: List[Instrument],
        bar_start: int,
        bar_end: Optional[int] = None,
        tokens: Optional[str] = None,
    ) -> str:
        """
        This method tokenizes a Musa (MIDI) object.

        Parameters
        ----------

        instruments: List[Instrument]
            the list of instruments to tokenize.

        Returns
        -------

        tokens: str
            the list of tokens corresponding to all the windows.
        """
        if tokens is None:
            tokens = ""

        # loop in instruments
        for inst_idx, inst in enumerate(instruments):
            tokens += "TRACK_START "
            tokens += f"INST={inst.program} "
            if self.args.track_density:
                tokens += f"DENSITY={len(inst.notes)} "
            # loop in bars
            if bar_end is None:
                bar_end = len(self.midi_object.bars)
            bars = self.midi_object.bars[bar_start:bar_end]
            tokens = self.tokenize_track_bars(
                bar_start, bars, int(inst.program), tokens
            )
            if inst_idx + 1 == len(instruments):
                tokens += "TRACK_END"
            else:
                tokens += "TRACK_END "
        return tokens

    def tokenize_track_bars(
        self,
        bar_start_idx: int,
        bars: List[Bar],
        program: int,
        tokens: Optional[str] = None,
    ) -> str:
        """
        This method tokenizes a given list of musicaiz bar objects.

        Parameters
        ----------

        bars: List[Bar]

        tokens: str
            the number of bars per track to tokenize.

        Returns
        -------

        tokens: str
            the tokens corresponding to the bars.
        """
        if tokens is None:
            tokens = ""

        # check valid time unit
        if self.args.time_unit not in VALID_TIME_UNITS:
            raise ValueError(f"Invalid time unit: {self.args.time_unit}")

        for b, bar in enumerate(bars, bar_start_idx):
            bar_start = bar.start_ticks
            bar_end = bar.end_ticks
            # sort notes by start_ticks
            notes = self.midi_object.get_notes_in_bars(b, b+1, int(program))

            tokens += "BAR_START "
            if len(notes) == 0:
                delta_symb = get_symbolic_duration(
                    bar_end - bar_start, True
                )
                delta_val = int(
                    NoteLengths[delta_symb].value / NoteLengths[self.args.time_unit].value
                )
                #tokens += f"TIME_DELTA={delta_val} " if delta_val != 0 else "TIME_DELTA=1 "
                if delta_val != 0: tokens += f"TIME_DELTA={delta_val} "
                tokens += "BAR_END "
                continue
            else:
                all_note_starts = [note.start_ticks for note in notes]
                all_note_ends = [note.end_ticks for note in notes]
                if notes[0].start_ticks - bar_start != 0:
                    delta_symb = get_symbolic_duration(
                        notes[0].start_ticks, True
                    )
                    delta_val = int(
                        NoteLengths[delta_symb].value / NoteLengths[self.args.time_unit].value
                    )
                    if delta_val != 0:
                        tokens += f"TIME_DELTA={delta_val} "

            all_time_events = all_note_starts + all_note_ends
            num_notes = len(all_note_starts)
            i = 0
            event_idx, note_idx = 0, 0
            event_idxs, diffs = [0], []
            while True:
                # The 1st note event will always be the 1st note on
                note_idx = event_idx % num_notes
                if event_idx < num_notes:
                    tokens += f"NOTE_ON={notes[note_idx].pitch} "
                    if self.args.velocity:
                        tokens += f"VELOCITY={notes[note_idx].velocity} "
                else:
                    tokens += f"NOTE_OFF={notes[note_idx].pitch} "

                if len(event_idxs) == len(all_time_events):
                    break

                diffs = [event - all_time_events[event_idx] for event in all_time_events]
                try:
                    time_delta = min(diff for i, diff in enumerate(diffs) if diff >= 0 and i not in event_idxs)
                except:
                    break
                if time_delta != 0:
                    delta_symb = get_symbolic_duration(
                        time_delta, True
                    )
                    delta_val = int(
                        NoteLengths[delta_symb].value / NoteLengths[self.args.time_unit].value
                    )
                    if delta_val != 0:
                        tokens += f"TIME_DELTA={delta_val} "

                list_indexes = [i for i, diff in enumerate(diffs) if diff == time_delta and i not in event_idxs]

                els_on = [el for el in list_indexes if el < num_notes]
                els_off = [el for el in list_indexes if el >= num_notes]
                if len(els_on) != 0 and len(els_off) != 0:
                    event_idx = min(els_off)
                elif len(els_on) == 0 and len(els_off) != 0:
                    event_idx = min(els_off)
                elif len(els_on) != 0 and len(els_off) == 0:
                    event_idx = min(els_on)
                i += 1
                event_idxs.append(event_idx)
            if notes[-1].end_ticks < bar_end:
                delta_symb = get_symbolic_duration(
                    bar_end - notes[-1].end_ticks, True
                )
                delta_val = int(
                    NoteLengths[delta_symb].value / NoteLengths[self.args.time_unit].value
                )
                if delta_val != 0:
                    tokens += f"TIME_DELTA={delta_val} "
            tokens += "BAR_END "
        return tokens

    @staticmethod
    def _split_tokens(
        piece_tokens: List[str],
        token_start: str,
        token_end: str
    ) -> List[List[str]]:
        instr_tokens = []
        for idx, token in enumerate(piece_tokens):
            key = token.split("=")[0]
            if key == token_start:
                new_instr = []
                for tok in piece_tokens[idx:]:
                    if tok != token_end:
                        new_instr.append(tok)
                    else:
                        new_instr.append(tok)
                        break
                instr_tokens.append(new_instr)
        return instr_tokens

    @classmethod
    def split_tokens_by_track(cls, piece_tokens: List[str]) -> List[List[str]]:
        """Split tokens list by instrument"""
        instr_tokens = cls._split_tokens(piece_tokens, "TRACK_START", "TRACK_END")
        return instr_tokens

    @classmethod
    def split_tokens_by_bar(cls, instr_tokens: List[str]) -> List[List[str]]:
        """Split tokens list by bars"""
        bar_tokens = cls._split_tokens(instr_tokens, "BAR_START", "BAR_END")
        return bar_tokens

    @classmethod
    def tokens_to_musa(
        cls,
        tokens: str,
        absolute_timing: bool = True,
        time_unit: str = "SIXTY_FOUR",
        time_sig: str = TimingConsts.DEFAULT_TIME_SIGNATURE.value,
        resolution: int = TimingConsts.RESOLUTION.value,
    ) -> Musa:

        # TODO: Support time sig. changes and absolte_timing=False

        _, ticks_bar = ticks_per_bar(time_sig, resolution)

        """Converts a str valid tokens sequence in Musa objects."""
        # Initialize midi file to write
        midi = Musa(file=None)
        midi.resolution = resolution
        midi.time_signature_changes = [
            {
                "time_sig": TimeSignature(time_sig),
                "ms": 0.0
            }
        ]
        midi.tempo_changes = [
            {
                "tempo": 120,
                "ms": 0.0
            }
        ]
        tokens_list = tokens.split(" ")
        instruments_tokens = cls.split_tokens_by_track(tokens_list)
        midi.instruments_progs = []
        for inst_idx, instr_tokens in enumerate(instruments_tokens):
            # First index in instr_tokens is the instr program
            # We just want the INST token in this loop
            midi.instruments.append(
                Instrument(
                    program=int(instr_tokens[1].split("=")[1]),
                    general_midi=True
                )
            )
            bar_tokens = cls.split_tokens_by_bar(instr_tokens)
            midi.instruments_progs.append(midi.instruments[-1].program)
            if inst_idx == 0:
                for bar_idx, bar in enumerate(bar_tokens):
                    bar_obj = Bar(
                        time_sig=TimeSignature(time_sig),
                        start = bar_idx * ticks_bar,
                        end = (bar_idx + 1) * ticks_bar,
                    )
                    midi.total_bars += 1
                    midi.bars.append(bar_obj)

            global_time_delta = 0
            for bar_idx, bar in enumerate(bar_tokens):
                if absolute_timing:
                    global_time_delta_ticks = bar_idx * ticks_bar
                else:
                    global_time_delta_ticks = 0
                global_time_delta = 0
                for idx, token in enumerate(bar):
                    if idx + 1 == len(bar):
                        break
                    key = token.split("=")[0]
                    if key == "TIME_DELTA":
                        global_time_delta += int(token.split("=")[1])
                    if key == "NOTE_ON":
                        pitch_value = token.split("=")[1]
                        # if we encode velocity we'll write its value in the notes
                        if bar[idx + 1].split("=")[0] == "VELOCITY":
                            vel = bar[idx + 1].split("=")[1]
                        else:
                            vel = 127
                        time_deltas = []
                        for tok in bar[idx:]:
                            tok_key = tok.split("=")[0]
                            if tok_key == "TIME_DELTA":
                                time_deltas.append(int(tok.split("=")[1]))
                            elif tok == ("NOTE_OFF=" + pitch_value):
                                # create Note object and append it to the bars ans instr.
                                if absolute_timing:
                                    global_time_delta = global_time_delta
                                duration = sum(time_deltas)
                                start_time = int(global_time_delta * NoteLengths[time_unit].ticks()) + global_time_delta_ticks
                                end_time = int((global_time_delta + duration) * NoteLengths[time_unit].ticks()) + global_time_delta_ticks
                                if end_time - start_time <= 0:
                                    continue
                                note = Note(
                                    pitch=int(pitch_value),
                                    start=start_time,
                                    end=end_time,
                                    velocity=int(vel),
                                    instrument_prog=midi.instruments[inst_idx].program,
                                    bar_idx=bar_idx
                                )
                                midi.notes.append(note)
                                break
                            else:
                                continue
        return midi

    @classmethod
    def get_pieces_tokens(cls, tokens: str):
        return cls._get_pieces_tokens(tokens, "PIECE_START")

    @classmethod
    def get_tokens_analytics(cls, tokens: str) -> Dict[str, int]:
        return cls._get_tokens_analytics(tokens, "NOTE_ON", "PIECE_START")
