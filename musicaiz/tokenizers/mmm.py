
from typing import Optional, List, Dict, Union, TextIO
from pathlib import Path
import argparse
import logging


from musicaiz.loaders import Musa
from musicaiz.structure import Note, Instrument, Bar
from musicaiz.tokenizers import EncodeBase
from musicaiz.rhythm import (
    TimingConsts,
    ticks_per_bar,
    get_symbolic_duration,
    NoteLengths,
    TimeSignature,
)
from musicaiz import wrappers
from musicaiz.utils import sort_notes


# time units available to tokenize
VALID_TIME_UNITS = ["SIXTEENTH", "THIRTY_SECOND", "SIXTY_FOUR", "HUNDRED_TWENTY_EIGHT"]


logger = logging.getLogger("mmm-tokenizer")
logging.basicConfig(level = logging.INFO)


class MMMTokenizer(EncodeBase):
    """
    This class presents methods to compute the Multi-Track Music Machine Encoding.

    Attributes
    ----------
    file: Optional[Union[str, TextIO, Path]] = None,
        genre: Optional[str] = None,
        subgenre: Optional[str] = None,
        chords: bool = False,
        key: bool = False,
    """

    def __init__(
        self,
        file: Optional[Union[str, TextIO, Path]] = None
    ):

        self.midi_object = Musa(
            file=file,
            structure="bars",
            absolute_timing=False,
            quantize=False,
            cut_notes=False
        )

    @classmethod
    @wrappers.timeis
    def tokenize_path(
        cls,
        path: Union[str, Path],
        output_file: str,
        tokens: str = "",
        windowing: bool = True,
        time_unit: str = "THIRTY_SECOND",
        num_programs: Optional[List[int]] = None,
        shuffle_tracks: bool = True,
        track_density: bool = False,
        window_size: int = 4,
        hop_length: int = 1,
    ):
        """
        Tokenizes the MIDI files inside the given path.

        Parameters
        ----------

        path: Union[str, Path]
            the path to tokenize.

        output_file: str
            the output file.

        windowing: bool
            if True, the method tokenizes each file by applying bars windowing.
        
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
        """
        text_file = open(output_file + ".txt", "w")

        tokens_list = wrappers.multiprocess_path(
            func=cls.tokenize_file,
            path=path,
            args=[
                tokens,
                windowing,
                time_unit,
                num_programs,
                shuffle_tracks,
                track_density,
                window_size,
                hop_length,
            ]
        )
        tokens = "".join([t for t in tokens_list])

        # write and close file
        text_file.write(tokens)
        text_file.close()

        logger.info(f"Written {text_file} file.")

    @staticmethod
    def tokenize_file(
        file: Union[str, Path],
        prev_tokens: str = "",
        windowing: bool = True,
        time_unit: str = "THIRTY_SECOND",
        num_programs: Optional[List[int]] = None,
        shuffle_tracks: bool = True,
        track_density: bool = False,
        window_size: int = 4,
        hop_length: int = 1,
        time_sig: bool = False,
        velocity: bool = False,
    ) -> str:

        """
        Tokenizes a MIDI file.

        Parameters
        ----------

        file: Union[str, TextIO]
            The input MIDI file.
        
        prev_tokens: str
            Previous tokens to the 1st TRACK_START token.
        
        output_file: str
            the output file.

        windowing: bool
            if True, the method tokenizes each file by applying bars windowing.
        
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
        
        velocity: bool = False

        Returns
        -------
        
        tokens: str
            The sequence ob tokens corresponding to the input file.
        """
        mmm_tok = MMMTokenizer(file)
        tokens = mmm_tok.tokenize_musa_obj(
            prev_tokens=prev_tokens,
            windowing=windowing,
            time_unit=time_unit,
            num_programs=num_programs,
            shuffle_tracks=shuffle_tracks,
            track_density=track_density,
            window_size=window_size,
            hop_length=hop_length,
            time_sig=time_sig,
            velocity=velocity,
        )
        return tokens
    
    def tokenize_musa_obj(
        self,
        prev_tokens: str = "",
        windowing: bool = True,
        time_unit: str = "SIXTY_FOUR",
        num_programs: Optional[List[int]] = None,
        shuffle_tracks: bool = True,
        track_density: bool = False,
        window_size: int = 4,
        hop_length: int = 1,
        time_sig: bool = False,
        velocity: bool = False,
    ) -> str:
        """
        This method tokenizes a Musa (MIDI) object.

        Parameters
        ----------

        prev_tokens: str
            Previous tokens to the 1st TRACK_START token.

        num_programs: List[int]
            the number of programs to tokenize. If None, the method
            tokenizes all the tracks.

        shuffle_tracks: bool
            shuffles the order of tracks in each window (PIECE).

        track_density: bool
            if True a token DENSITY is added at the beggining of each track.
            The token DENSITY is the total notes of the track or instrument.

        window_size: int
            the number of bars per track to tokenize.

        hop_length: int
            the number of bars to slice when tokenizing.
            If a MIDI file contains 5 bars and the window size is 4 and the hop length is 1,
            it'll be splitted in 2 PIECE tokens, one from bar 1 to 4 and the other on from
            bar 2 to 5 (somehow like audio FFT).
        
        velocity: bool = False

        Returns
        -------

        all_tokens: List[str]
            the list of tokens corresponding to all the windows.
        """
        # Do not tokenize the tracks that are not in num_programs
        # but if num_programs is None then tokenize all instruments

        tokenized_instruments = []
        if num_programs is not None:
            for inst in self.midi_object:
                if inst.program in num_programs:
                    tokenized_instruments.append(inst)
        else:
            tokenized_instruments = self.midi_object.instruments

        if not windowing:
            if time_sig:
                time_sig_tok = f"TIME_SIG={self.midi_object.time_sig.time_sig} "
            else:
                time_sig_tok = ""
            tokens = self.tokenize_tracks(
                instruments=tokenized_instruments,
                bar_start=0,
                tokens="PIECE_START " + prev_tokens + " " + time_sig_tok,
                track_density=track_density,
                time_unit=time_unit,
                velocity=velocity
            )
            tokens += "\n"
        else:
            # Now tokenize and create a PIECE for each window
            # that is defined in terms of bars
            # loop in bars
            tokens = ""
            for i in range(0, self.midi_object.total_bars, hop_length):
                if i + window_size == self.midi_object.total_bars:
                    break
                if time_sig:
                    time_sig_tok = f"TIME_SIG={self.midi_object.time_sig.time_sig} "
                else:
                    time_sig_tok = ""
                tokens += self.tokenize_tracks(
                    tokenized_instruments,
                    bar_start=i,
                    bar_end=i+window_size,
                    tokens="PIECE_START " + prev_tokens + " " + time_sig_tok,
                    time_unit=time_unit,
                    track_density=track_density,
                    velocity=velocity
                )
                tokens += "\n"
        return tokens

    @classmethod
    def tokenize_tracks(
        cls,
        instruments: List[Instrument],
        bar_start: int,
        bar_end: Optional[int] = None,
        tokens: Optional[str] = None,
        time_unit: str = "SIXTY_FOUR",
        track_density: bool = False,
        velocity: bool = False
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
            if track_density:
                tokens += f"DENSITY={len(inst.notes)} "
            # loop in bars
            if bar_end is None:
                bar_end = len(inst.bars)
            bars = inst.bars[bar_start:bar_end]
            tokens = cls.tokenize_track_bars(bars, tokens, time_unit, velocity=velocity)
            if inst_idx + 1 == len(instruments):
                tokens += "TRACK_END"
            else:
                tokens += "TRACK_END "
        return tokens

    @staticmethod
    def tokenize_track_bars(
        bars: List[Bar],
        tokens: Optional[str] = None,
        time_unit: str = "SIXTY_FOUR",
        velocity: bool = False
    ) -> str:
        """
        This method tokenizes a given list of musicaiz bar objects.

        Parameters
        ----------

        bars: List[Bar]

        tokens: str
            the number of bars per track to tokenize.
        
        time_unit: str = "SIXTY_FOUR"

        time_sig: str = TimingConsts.DEFAULT_TIME_SIGNATURE.value

        velocity: bool = False
            If we want to include the token `VELOCITY` (not included in the original MMM encoding paper.)

        Returns
        -------

        tokens: str
            the tokens corresponding to the bars.
        """
        if tokens is None:
            tokens = ""

        # check valid time unit
        if time_unit not in VALID_TIME_UNITS:
            raise ValueError(f"Invalid time unit: {time_unit}")

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
                    NoteLengths[delta_symb].value / NoteLengths[time_unit].value
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
                        NoteLengths[delta_symb].value / NoteLengths[time_unit].value
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
                    if velocity:
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
                        NoteLengths[delta_symb].value / NoteLengths[time_unit].value
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
                    NoteLengths[delta_symb].value / NoteLengths[time_unit].value
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file to write",
        required=True,
    )
    parser.add_argument(
        "--path",
        type=str,
        help="path with midi files to tokenize",
        required=True,
    )
    parser.add_argument(
        "--windowing",
        type=bool,
        help="Applies windowing to tokenize the files",
        required=True,
    )
    parser.add_argument(
        "--window_size",
        type=int,
        help="the bars of the midi file to tokenize for each PIECE token",
        required=False,
        default=4,
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        help="the slicing to apply to the midi file to tokenize",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--num_programs",
        type=List[int],
        help="Program numbers of the tracks to tokenize",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--shuffle_tracks",
        type=bool,
        help="shuffles the tracks if true and if windowing is also true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--track_density",
        type=bool,
        help="adds the note density token",
        required=False,
        default=False,
    )
    return parser.parse_args()
    

if __name__ == "__main__":
    # use the arg parser to get the tokenize
    args = parse_args()
    MMMTokenizer.tokenize_path(
        output_file=args.output_file,
        path=args.path,
        windowing=args.windowing,
        window_size=args.window_size,
        hop_length=args.hop_length,
        num_programs=args.num_programs,
        shuffle_tracks=args.shuffle_tracks,
        track_density=args.track_density,
    )
