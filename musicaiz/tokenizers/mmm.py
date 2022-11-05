
from typing import Optional, List, Dict, Union, TextIO
from pathlib import Path
import argparse
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
from musicaiz.utils import sort_notes


# time units available to tokenize
VALID_TIME_UNITS = ["SIXTEENTH", "THIRTY_SECOND", "SIXTY_FOUR", "HUNDRED_TWENTY_EIGHT"]


logger = logging.getLogger("mmm-tokenizer")
logging.basicConfig(level = logging.INFO)


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
            raise ValueError(f"No `MMMTokenizerArguments` passed.")
        self.args = args

        # Convert file into a Musa object to be processed
        if file is not None:
            self.midi_object = Musa(
                file=file,
                structure="bars",
                absolute_timing=False,
                quantize=self.args.quantize,
                cut_notes=False
            )
    
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
                time_sig_tok = f"TIME_SIG={self.midi_object.time_sig.time_sig} "
            else:
                time_sig_tok = ""
            tokens = self.tokenize_tracks(
                instruments=tokenized_instruments,
                bar_start=0,
                tokens="PIECE_START " + self.args.prev_tokens + " " + time_sig_tok,
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
                    time_sig_tok = f"TIME_SIG={self.midi_object.time_sig.time_sig} "
                else:
                    time_sig_tok = ""
                tokens += self.tokenize_tracks(
                    tokenized_instruments,
                    bar_start=i,
                    bar_end=i+self.args.window_size,
                    tokens="PIECE_START " + self.args.prev_tokens + " " + time_sig_tok,
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

        track_density: bool
            if True a token DENSITY is added at the beggining of each track.
            The token DENSITY is the total notes of the track or instrument.
        
        velocity: bool = False

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
                bar_end = len(inst.bars)
            bars = inst.bars[bar_start:bar_end]
            tokens = self.tokenize_track_bars(
                bars, tokens
            )
            if inst_idx + 1 == len(instruments):
                tokens += "TRACK_END"
            else:
                tokens += "TRACK_END "
        return tokens

    def tokenize_track_bars(
        self,
        bars: List[Bar],
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

        for bar in bars:
            bar_start = bar.start_ticks
            bar_end = bar.end_ticks
            # sort notes by start_ticks
            bar.notes = sort_notes(bar.notes)            
            all_note_starts = [note.start_ticks for note in bar.notes]
            all_note_ends = [note.end_ticks for note in bar.notes]

            tokens += "BAR_START "
            if len(bar.notes) == 0:
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
                if bar.notes[0].start_ticks - bar_start != 0:
                    delta_symb = get_symbolic_duration(
                        bar.notes[0].start_ticks, True
                    )
                    delta_val = int(
                        NoteLengths[delta_symb].value / NoteLengths[self.args.time_unit].value
                    )
                    #tokens += f"TIME_DELTA={delta_val - bar_start} " if delta_val - bar_start != 0 else "TIME_DELTA=1 "
                    if delta_val - bar_start != 0: tokens += f"TIME_DELTA={delta_val - bar_start} "
            
            all_time_events = all_note_starts + all_note_ends
            num_notes = len(all_note_starts)
            i = 0
            event_idx, note_idx = 0, 0
            event_idxs, diffs = [0], []
            while True:
                # The 1st note event will always be the 1st note on
                note_idx = event_idx % num_notes
                if event_idx < num_notes:
                    tokens += f"NOTE_ON={bar.notes[note_idx].pitch} "
                    if self.args.velocity:
                        tokens += f"VELOCITY={bar.notes[note_idx].velocity} "
                else:
                    tokens += f"NOTE_OFF={bar.notes[note_idx].pitch} "
                
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
                    #tokens += f"TIME_DELTA={delta_val} " if delta_val != 0 else "TIME_DELTA=1 "
                    if delta_val != 0: tokens += f"TIME_DELTA={delta_val} "

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
            if bar.notes[-1].end_ticks < bar_end:
                delta_symb = get_symbolic_duration(
                        bar_end - bar.notes[-1].end_ticks, True
                )
                delta_val = int(
                    NoteLengths[delta_symb].value / NoteLengths[self.args.time_unit].value
                )
                #tokens += f"TIME_DELTA={delta_val} " if delta_val != 0 else "TIME_DELTA=1 "
                if delta_val != 0: tokens += f"TIME_DELTA={delta_val} "
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
        tokens: List[str],
        absolute_timing: bool = True,
        time_unit: str = "SIXTY_FOUR",
        time_sig: str = TimingConsts.DEFAULT_TIME_SIGNATURE.value,
        resolution: int = TimingConsts.RESOLUTION.value,
    ) -> Musa:

        # TODO: Support time sig. changes

        if absolute_timing:
            _, ticks_bar = ticks_per_bar(time_sig, resolution)

        """Converts a str valid tokens sequence in Musa objects."""
        # Initialize midi file to write
        midi = Musa()
        midi.time_sig = TimeSignature(time_sig)

        instruments_tokens = cls.split_tokens_by_track(tokens)
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
            global_time_delta = 0
            for bar_idx, bar in enumerate(bar_tokens):
                bar_obj = Bar()
                midi.instruments[inst_idx].bars.append(bar_obj)
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
                                )
                                midi.instruments[inst_idx].bars[bar_idx].notes.append(note)
                                break
                            else:
                                continue
        return midi

    @staticmethod
    def _get_pieces_tokens(tokens: str) -> List[List[str]]:
        """Converts the tokens str that can contain one or more
        pieces into a list of pieces that are also lists which contain
        one item per token.

        Example
        -------
        >>> tokens = "PIECE_START INST=0 ... PIECE_START ..."
        >>> dataset_tokens = _get_pieces_tokens(tokens)
        >>> [
                ["PIECE_START INST=0 ...],
                ["PIECE_START ...],
            ]
        """
        tokens = tokens.split("PIECE_START")
        tokens.remove("")
        dataset_tokens = []
        for piece in tokens:
            piece_tokens = piece.split(" ")
            piece_tokens.remove("")
            dataset_tokens.append(piece_tokens)
        return dataset_tokens

    @classmethod
    def get_tokens_analytics(cls, tokens: str) -> Dict[str, int]:
        """
        Extracts features to aanlyze the given token sequence.

        Parameters
        ----------

        tokens: str
            A token sequence.

        Returns
        -------

        analytics: Dict[str, int]
            The ``analytics`` dict keys are:
                - ``total_tokens``
                - ``unique_tokens``
                - ``total_notes``
                - ``unique_notes``
                - ``total_bars``
                - ``total_instruments``
                - ``unique_instruments``
        """
        # Convert str in list of pieces that contain tokens
        dataset_tokens = cls._get_pieces_tokens(tokens)
        # Start the analysis
        note_counts, bar_counts, instr_counts = 0, 0, 0  # total notes and bars (also repeated note values)
        total_toks = 0
        unique_tokens, unique_notes, unique_instr = [], [], []  # total non-repeated tokens
        unique_genres, unique_composers, unique_periods = [], [], []
        for piece, toks in enumerate(dataset_tokens):
            for tok in toks:
                total_toks += 1
                if tok not in unique_tokens:
                    unique_tokens.append(tok)
                if "NOTE_ON" in tok:
                    note_counts += 1
                if "BAR_START" in tok:
                    bar_counts += 1
                if "INST" in tok:
                    instr_counts += 1
                if "NOTE_ON" in tok and tok not in unique_notes:
                    unique_notes.append(tok)
                if "INST" in tok and tok not in unique_instr:
                    unique_instr.append(tok)
                if "GENRE" in tok and tok not in unique_genres:
                    unique_genres.append(tok)
                if "PERIOD" in tok and tok not in unique_periods:
                    unique_periods.append(tok)
                if "COMPOSER" in tok and tok not in unique_composers:
                    unique_composers.append(tok)

        analytics = {
            "total_pieces": piece + 1,
            "total_tokens": total_toks,
            "unique_tokens": len(unique_tokens),
            "total_notes": note_counts,
            "unique_notes": len(unique_notes),
            "total_bars": bar_counts,
            "total_instruments": instr_counts,
        }
        if len(unique_genres) != 0:
            analytics.update({"unique_genres": len(unique_genres)})
        if len(unique_periods) != 0:
            analytics.update({"unique_periods": len(unique_periods)})
        if len(unique_composers) != 0:
            analytics.update({"unique_composers": len(unique_composers)})

        return analytics
