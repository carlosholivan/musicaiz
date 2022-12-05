
from typing import Optional, List, Dict, Union, TextIO
from pathlib import Path
import logging
from dataclasses import dataclass


from musicaiz.loaders import Musa
from musicaiz.rhythm.timing import Subdivision, TimeSignature
from musicaiz.structure import Note, Instrument, Bar
from musicaiz.tokenizers import EncodeBase, TokenizerArguments
from musicaiz.rhythm import (
    TimingConsts,
    NoteLengths,
    ms_per_note,
)


# time units available to tokenize
VALID_TIME_UNITS = ["SIXTEENTH", "THIRTY_SECOND", "SIXTY_FOUR", "HUNDRED_TWENTY_EIGHT"]


logger = logging.getLogger("remi-tokenizer")
logging.basicConfig(level=logging.INFO)


@dataclass
class REMITokenizerArguments(TokenizerArguments):
    """
    This is the REMI arguments class.
    The default parameters are selected by following the original REMI representation.

    prev_tokens: str
        if we want to add tokens after the `PIECE_START` token and before
        the 1st TRACK_START token (for conditioning...).

    sub_beat: str
        the note length in `VALID_TIME_UNITS` that one bar is divided in. Note that
        this refers to the subdivisions of a 4/4 bar, so if we have different time
        signatures, the ratio of `sub_beat / bar.denominator` will be maintained
        to prevent wrong subdivisions when using bars with different denominators.

    velocity: bool
        if we want to add the velocity token. Velocities ranges between 1 and 128 (ints).

    quantize: bool
        if we want to quantize the symbolic music data for tokenizing.
    """

    prev_tokens: str = ""
    sub_beat: str = "SIXTEENTH"  # 16 in a 4/4 bar
    num_programs: Optional[List[int]] = None
    velocity: bool = False
    quantize: bool = True


class REMITokenizer(EncodeBase):
    """
    This class presents methods to compute the REMI Encoding.
    The REMI encoding for piano pieces (mono-track) was introduced in:
    *Huang, Y. S., & Yang, Y. H. (2020, October).
    Pop music transformer: Beat-based modeling and generation of expressive pop piano compositions.
    In Proceedings of the 28th ACM International Conference on Multimedia (pp. 1180-1188).*

    For multi-track pieces, the REMI encoding was adapted by:
    *Zeng, M., Tan, X., Wang, R., Ju, Z., Qin, T., & Liu, T. Y. (2021).
    Musicbert: Symbolic music understanding with large-scale pre-training.
    arXiv preprint arXiv:2106.05630.*

    In this implementation, both mono-track and multi-track are handled.

    This encoding works divides a X/4 bar in 16 sub-beats which means that
    each quarter or crotchet is divided in 4 sub-beats (16th notes). In spite
    of that and for allowing developers having more control over the beats
    division, we can change that value to other divisions as a function of the
    selected note length.
    The music is quantized but, as happens with the sub-beats tokens, we can
    specify if we want to quantize or not with the `quantize` argument.
    The note's duration are ex`ressed in its symbolic length, e.g., a duration
    equal to 1 is a whole note and a duration of 16 is a 16th note.

    This hiherarchical tokenization is organized as follows:
        - Bar -> [BAR] Position from 1/16 to 16/16
        - Position -> [POS=1/16] [TEMPO=X] [INST=X] [PITCH=X] [DUR=1] [VEL=X] ...

    Note that if a position or sub-beat does not contain notes, it'll not be present
    in the tokenization. This allows preventing having usueful  or "empty" tokens.

    Attributes
    ----------
    file: Optional[Union[str, TextIO, Path]] = None
    """

    def __init__(
        self,
        file: Union[str, TextIO, Path],
        args: REMITokenizerArguments = None
    ):

        if args is None:
            raise ValueError("No `REMITokenizerArguments` passed.")
        self.args = args

        # Convert file into a Musa object to be processed
        self.midi_object = Musa(
            file=file,
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

        tokens = self.tokenize_bars(
            tokens=self.args.prev_tokens,
        )
        tokens += "\n"
        return tokens

    def tokenize_bars(
        self,
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
        if self.args.sub_beat not in VALID_TIME_UNITS:
            raise ValueError(f"Invalid time unit: {self.args.sub_beat}")

        prev_bpm = 0
        for b_idx, bar in enumerate(self.midi_object.bars):
            all_notes = self.midi_object.get_notes_in_bar(b_idx)
            if len(all_notes) == 0:
                continue
            tokens += f"BAR={b_idx} "
            tokens += f"TIME_SIG={bar.time_sig.time_sig} "
            # Get subdivisions in bar with bar index
            subdivs = self.midi_object.get_subbeats_in_bar(bar_idx=b_idx)
            subdivs_idxs = [i for i in range(len(subdivs))]
            for sub in subdivs_idxs:
                notes = self.midi_object.get_notes_in_subbeat_bar(
                    sub, b_idx
                )
                if len(notes) == 0:
                    continue
                bpm = subdivs[sub].bpm
                tokens += f"SUB_BEAT={sub} "
                if bpm != prev_bpm:
                    tokens += f"TEMPO={int(bpm)} "
                prev_bpm = bpm
                # Get notes in subdivision with subdivision min and max indexes
                prev_prog = notes[0].instrument_prog
                for note in notes:
                    prog = note.instrument_prog
                    if prog != prev_prog or "PITCH" not in tokens:
                        tokens += f"INST={note.instrument_prog} "
                    tokens += f"PITCH={note.pitch} "
                    note_dur = int(
                        NoteLengths[note.symbolic].value / NoteLengths[self.args.sub_beat].value
                    )
                    tokens += f"DUR={note_dur} "
                    tokens += f"VELOCITY={note.velocity} "
                    prev_prog = prog
                prev_bpm = bpm
        return tokens.rstrip()

    @staticmethod
    def _split_tokens(
        piece_tokens: List[str],
        token: str,
    ) -> List[List[str]]:
        """Split tokens list by token"""
        indices = [i for i, x in enumerate(piece_tokens) if x.split("=")[0] == token]
        lst = [piece_tokens[start:end] for start, end in zip([0, *indices], [*indices, len(piece_tokens)])]
        return [el for el in lst if el != []]

    @classmethod
    def split_tokens_by_bar(
        cls,
        piece_tokens: List[str],
    ) -> List[List[str]]:
        """Split tokens list by bar"""
        return cls._split_tokens(piece_tokens, "BAR")

    @classmethod
    def split_tokens_by_subbeat(
        cls,
        piece_tokens: List[str],
    ) -> List[List[str]]:
        """Split tokens list by subbeat"""
        bars = cls.split_tokens_by_bar(piece_tokens)
        subbeats = []
        for bar in bars:
            subbeats.extend(cls._split_tokens(bar, "SUB_BEAT"))
        return subbeats

    # TODO
    @classmethod
    def tokens_to_musa(
        cls,
        tokens: Union[str, List[str]],
        sub_beat: str = "SIXTY_FOUR", 
        resolution: int = TimingConsts.RESOLUTION.value,
    ) -> Musa:

        """Converts a str valid tokens sequence in Musa objects.

        This representation does not store beats in the Musa object, but
        it stores bars, notes, instruments and subbeats."""
        # Initialize midi file to write
        midi = Musa(file=None, resolution=resolution)
        midi.resolution = resolution

        if isinstance(tokens, str):
            tokens = tokens.split(" ")

        midi.subbeats = []
        midi.bars = []
        midi.instruments_progs = []

        sb_tokens = cls.split_tokens_by_subbeat(tokens)
        global_subbeats, pos, total_subbeats = 0, 0, 0
        prev_pos, prev_bar_pos = 0, 0
        for sb_tokens in sb_tokens:
            if "BAR" in sb_tokens[0]:
                time_sig = sb_tokens[1].split("=")[1]
                bar_pos = int(sb_tokens[0].split("=")[1])
                # if bar has incomplete subbeats, we fill them
                prev_bar_subbeats = len([s for s in midi.subbeats if s.bar_idx == prev_bar_pos])
                if prev_bar_subbeats != 0:
                    for ii in range(prev_bar_subbeats, int(TimeSignature(time_sig)._notes_per_bar(sub_beat))):
                        midi.subbeats.append(
                            Subdivision(
                                bpm=tempo_token,
                                resolution=midi.resolution,
                                start=(midi.subbeats[-1].global_idx + 1) * sec_subbeat,
                                end=((midi.subbeats[-1].global_idx + 1) + 1) * sec_subbeat,
                                time_sig=TimeSignature(time_sig=time_sig),
                                global_idx=global_subbeats + ii,
                                bar_idx=prev_bar_pos,
                                beat_idx=None,
                            )
                        )
                # empty bars
                if bar_pos != prev_bar_pos + 1 and bar_pos != 0:
                    # Generate all subdivisions of the empty bar
                    for b in range(prev_bar_pos + 1, bar_pos):
                        total_subbeats += (b - prev_bar_pos - 1) * int(TimeSignature(time_sig)._notes_per_bar(sub_beat))
                        for ii in range(0, int(TimeSignature(time_sig)._notes_per_bar(sub_beat))):
                            midi.subbeats.append(
                                Subdivision(
                                    bpm=tempo_token,
                                    resolution=midi.resolution,
                                    start=(total_subbeats + ii) * sec_subbeat,
                                    end=((total_subbeats + ii) + 1) * sec_subbeat,
                                    time_sig=TimeSignature(time_sig=time_sig),
                                    global_idx=total_subbeats + ii,
                                    bar_idx=b,
                                    beat_idx=None,
                                )
                            )
                        total_subbeats += int(TimeSignature(time_sig)._notes_per_bar(sub_beat))
                global_subbeats = total_subbeats
                total_subbeats += int(TimeSignature(time_sig)._notes_per_bar(sub_beat))
                prev_bar_pos = bar_pos
                continue
            if "SUB_BEAT" in sb_tokens[0]:
                if sb_tokens[1].split("=")[0] == "TEMPO":
                    tempo_token = int(sb_tokens[1].split("=")[1])
                if sb_tokens[2].split("=")[0] == "INST":
                    inst_token = int(sb_tokens[2].split("=")[1])
                    if inst_token not in midi.instruments_progs:
                        midi.instruments_progs.append(inst_token)
                        midi.instruments.append(
                            Instrument(
                                program=inst_token
                            )
                        )
                pos = int(sb_tokens[0].split("=")[1])
                # Last subbeat
                sec_subbeat = ms_per_note(
                    sub_beat.lower(),
                    tempo_token,
                    midi.resolution
                ) / 1000
                # Generate beats that have no notes
                empty_subbeats = pos - prev_pos - 1
                if empty_subbeats >= 1:
                    for ii in range(prev_pos, pos):
                        midi.subbeats.append(
                            Subdivision(
                                bpm=tempo_token,
                                resolution=midi.resolution,
                                start=(global_subbeats + ii) * sec_subbeat,
                                end=((global_subbeats + ii) + 1) * sec_subbeat,
                                time_sig=TimeSignature(time_sig=time_sig),
                                global_idx=global_subbeats + ii,
                                bar_idx=bar_pos,
                                beat_idx=None,
                            )
                        )
                sb = Subdivision(
                    bpm=tempo_token,
                    resolution=midi.resolution,
                    start=(midi.subbeats[-1].global_idx + 1) * sec_subbeat,
                    end=((midi.subbeats[-1].global_idx + 1) + 1) * sec_subbeat,
                    time_sig=TimeSignature(time_sig=time_sig),
                    global_idx=midi.subbeats[-1].global_idx + 1,
                    bar_idx=bar_pos,
                    beat_idx=None,
                )
                midi.subbeats.append(sb)
                prev_pos = pos + 1

                for j, tok in enumerate(sb_tokens):
                    if tok.split("=")[0] == "PITCH":
                        pitch = int(tok.split("=")[1])
                        dur = int(sb_tokens[j + 1].split("=")[1])
                        vel = int(sb_tokens[j + 2].split("=")[1])
                        note = Note(
                            start=sb.start_sec,
                            end=sb.start_sec + dur * sec_subbeat,
                            instrument_prog=int(inst_token),
                            pitch=pitch,
                            velocity=vel,
                            subbeat_idx=sb.global_idx,
                        )
                        midi.notes.append(note)
        # complete subbeats last bar if they're incomplete
        prev_bar_subbeats = len([s for s in midi.subbeats if s.bar_idx == bar_pos])
        if prev_bar_subbeats != 0:
            for ii in range(prev_bar_subbeats, int(TimeSignature(time_sig)._notes_per_bar(sub_beat))):
                midi.subbeats.append(
                    Subdivision(
                        bpm=tempo_token,
                        resolution=midi.resolution,
                        start=(midi.subbeats[-1].global_idx + 1) * sec_subbeat,
                        end=((midi.subbeats[-1].global_idx + 1) + 1) * sec_subbeat,
                        time_sig=TimeSignature(time_sig=time_sig),
                        global_idx=midi.subbeats[-1].global_idx + 1,
                        bar_idx=bar_pos,
                        beat_idx=None,
                    )
                )
        # Generate Bars
        bar_pos = 0
        last_bar = midi.subbeats[-1].bar_idx
        for i in range(0, last_bar + 1):
            sub = [s for s in midi.subbeats if s.bar_idx == i][0]
            end_sec = [s.end_sec for s in midi.subbeats if s.bar_idx == i][-1]
            midi.bars.append(
                Bar(
                    start=sub.start_sec,
                    end=end_sec,
                    time_sig=sub.time_sig,
                    resolution=resolution,
                    bpm=sub.bpm
                )
            )
        return midi

    @classmethod
    def get_tokens_analytics(cls, tokens: str) -> Dict[str, int]:
        return cls._get_tokens_analytics(tokens, "PITCH", "BAR=0")
