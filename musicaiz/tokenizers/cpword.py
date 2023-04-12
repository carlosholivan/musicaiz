from typing import Optional, Union, TextIO, List
from pathlib import Path
from musicaiz.rhythm import NoteLengths, TimingConsts, TimeSignature, Subdivision, ms_per_note
from dataclasses import dataclass
from musicaiz.tokenizers import TokenizerArguments, EncodeBase
from musicaiz.loaders import Musa
from musicaiz.structure import Note, Instrument, Bar
import copy


VALID_TIME_UNITS = ["SIXTEENTH", "THIRTY_SECOND", "SIXTY_FOUR", "HUNDRED_TWENTY_EIGHT"]

@dataclass
class CPWordTokenizerArguments(TokenizerArguments):

    prev_tokens: str = ""
    sub_beat: str = "SIXTEENTH"  # 16 in a 4/4 bar
    num_programs: Optional[List[int]] = None
    velocity: bool = False
    quantize: bool = True
    rest: bool = False
    chord: bool = False


class CPWordTokenizer(EncodeBase):

    def __init__(
        self,
        file: Union[str, TextIO, Path],
        args: CPWordTokenizerArguments = None
    ):

        if args is None:
            raise ValueError("No `CPWordTokenizerArguments` passed.")
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
        tokens: str = "",
    ) -> str:
        """
        This method tokenizes one instrument from a Musa (MIDI) object.
        Parameters
        ----------
        inst_prog: int
            the number of the instrument program
        tokens: str
            the number of bars per track to tokenize.
        Returns
        -------
        tokens: str
            the tokens corresponding to the instrument.
        """

        for b_idx in range(len(self.midi_object.bars)):
            tokens += self.tokenize_bar(b_idx=b_idx)

        tokens = tokens[:-1]
        #tokens += "\n"

        return tokens

    def tokenize_bar(
        self,
        b_idx: int,
        tokens: str = "",
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
        bar = self.midi_object.bars[b_idx]
        notes = self.midi_object.get_notes_in_bar(
            bar_idx=b_idx, 
            program=self.args.num_programs
        ) 

        if len(notes) != 0:
            family = "METRIC"
            tokens += f"FAMILY={family} "
            tokens += f"BAR={b_idx} "
            tokens += "PITCH=NONE "
            tokens += "VELOCITY=NONE "
            tokens += "DURATION=NONE "
            tokens += "PROGRAM=NONE "
            tokens += "TEMPO=NONE "
            tokens += f"TIME_SIG={bar.time_sig.time_sig} "
            if self.args.chord is True:
                tokens += "CHORD=NONE "
            if self.args.rest is True:
                tokens += "REST=NONE "
        else:
            return ""

        # Get subdivisions in bar with bar index
        subdivs = self.midi_object.get_subbeats_in_bar(bar_idx=b_idx)
        subdivs_idxs = [i for i in range(len(subdivs))]
        for sub in subdivs_idxs:
            tokens += self.tokenize_position(subdivision=subdivs[sub])

        return tokens

    def tokenize_position(
            self,
            subdivision,
        ) -> str:
        """
        This method tokenizes a given subdivision of musicaiz.
        Parameters
        ----------
        subdivision
        tokens: str
            the number of bars per track to tokenize.
        Returns
        -------
        tokens: str
            the tokens corresponding to the bars.
        """

        # check valid time unit
        if self.args.sub_beat not in VALID_TIME_UNITS:
            raise ValueError(f"Invalid time unit: {self.args.sub_beat}")

        # Get notes in subdivision
        notes = self.midi_object.get_notes_in_subbeat(
            subbeat_idx=subdivision.global_idx,
            program=self.args.num_programs
        )

        if len(notes) != 0:
            family = "METRIC"
            tokens = f"FAMILY={family} "
            tokens += f"POSITION={int(subdivision.global_idx)} "
            tokens += "PITCH=NONE "
            tokens += "VELOCITY=NONE "
            tokens += "DURATION=NONE "
            tokens += "PROGRAM=NONE "
            bpm = subdivision.bpm
            tokens += f"TEMPO={int(bpm)} "
            tokens += "TIME_SIG=NONE "
            if self.args.chord is True:
                tokens += "CHORD=NONE "
            if self.args.rest is True:
                tokens += "REST=NONE "
        else:
            return ""

        for note in notes:
            family = "NOTE"
            tokens += f"FAMILY={family} "
            tokens += "POSITION=NONE "
            tokens += f"PITCH={note.pitch} "
            tokens += f"VELOCITY={note.velocity} "
            note_dur = int(NoteLengths[note.symbolic].value / NoteLengths[self.args.sub_beat].value)
            if note_dur == 0:
                print(NoteLengths[note.symbolic].value, note.instrument_prog)
            tokens += f"DURATION={note_dur} "
            tokens += f"PROGRAM={int(note.instrument_prog)} "
            tokens += "TEMPO=NONE "
            tokens += "TIME_SIG=NONE "
            if self.args.chord is True:
                tokens += "CHORD=NONE "
            if self.args.rest is True:
                tokens += "REST=NONE "

        return tokens
    
    @staticmethod
    def _split_tokens(
        piece_tokens: List[str],
        token: str,
    ) -> List[List[str]]:
        """Split tokens list by token"""
        indices = [(i - 1) for i, x in enumerate(piece_tokens) if x.split("=")[0] == token]
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
            subbeats.extend(cls._split_tokens(bar, "POSITION"))
        return subbeats

    @classmethod
    def tokens_to_musa(
        cls,
        tokens: Union[str, List[str]],
        sub_beat: str = "SIXTEENTH", 
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
            if "BAR" in sb_tokens[1]:
                time_sig = sb_tokens[7].split("=")[1]
                bar_pos = int(sb_tokens[1].split("=")[1])
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
                prev_pos = total_subbeats
                global_subbeats = total_subbeats
                total_subbeats += int(TimeSignature(time_sig)._notes_per_bar(sub_beat))
                prev_bar_pos = bar_pos
                continue
            if "POSITION" in sb_tokens[1]:
                if "NONE" not in sb_tokens[1]:
                    tempo_token = int(sb_tokens[6].split("=")[1])
                    pos = int(sb_tokens[1].split("=")[1])
                    # Last subbeat
                    sec_subbeat = ms_per_note(
                        sub_beat.lower(),
                        tempo_token,
                        midi.resolution
                    ) / 1000

                    #TEMPO changes
                    if tempo_token not in midi.tempo_changes:
                        midi.tempo_changes.append(
                            {
                                "tempo": tempo_token,
                                "ms": pos * sec_subbeat * 1000,
                            }
                        )
                    num_subbeats_bar = int(TimeSignature(time_sig)._notes_per_bar(sub_beat))
                    num_beats_bar = int(TimeSignature(time_sig).num)
                    num_subbeats_beat = num_subbeats_bar / num_beats_bar

                    beat_idx = pos // num_beats_bar

                    # Generate beats that have no notes
                    empty_subbeats = pos - prev_pos - 1
                    if empty_subbeats >= 1:
                        for ii in range(prev_pos, pos):
                            midi.subbeats.append(
                                Subdivision(
                                    bpm=tempo_token,
                                    resolution=midi.resolution,
                                    start=ii * sec_subbeat,
                                    end=(ii + 1) * sec_subbeat,
                                    time_sig=TimeSignature(time_sig=time_sig),
                                    global_idx=ii,
                                    bar_idx=bar_pos,
                                    beat_idx=None,
                                )
                            )
                    
                    sb = Subdivision(
                        bpm=tempo_token,
                        resolution=midi.resolution,
                        start=pos * sec_subbeat,
                        end=(pos + 1) * sec_subbeat,
                        time_sig=TimeSignature(time_sig=time_sig),
                        global_idx=pos,
                        bar_idx=bar_pos,
                        beat_idx=beat_idx,
                    )
                    midi.subbeats.append(sb)
                    prev_pos = pos + 1
                else: #FAMILY=NOTE
                    inst_token = int(sb_tokens[5].split("=")[1])
                    if inst_token not in midi.instruments_progs:
                        midi.instruments_progs.append(inst_token)
                        midi.instruments.append(
                            Instrument(
                                program=inst_token
                            )
                        )
                    pitch = int(sb_tokens[2].split("=")[1])
                    vel = int(sb_tokens[3].split("=")[1])
                    dur = int(sb_tokens[4].split("=")[1])
                    note = Note(
                        start=sb.start_sec,
                        end=sb.start_sec + dur * sec_subbeat,
                        instrument_prog=inst_token,
                        pitch=pitch,
                        velocity=vel,
                        subbeat_idx=sb.global_idx,
                        bar_idx=bar_pos,
                        beat_idx=beat_idx,
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
            subs =[s for s in midi.subbeats if s.bar_idx == i]
            sub = [s for s in midi.subbeats if s.bar_idx == i][0]
            end_sec = [s.end_sec for s in midi.subbeats if s.bar_idx == i][-1]

            midi.bars.append(
                Bar(
                    start=subs[0].start_sec,
                    end=subs[-1].end_sec,
                    time_sig=sub.time_sig,
                    resolution=resolution,
                    bpm=sub.bpm
                )
            )
        midi.total_bars = len(midi.bars)
        return midi
