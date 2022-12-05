from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Musa(_message.Message):
    __slots__ = ["absolute_timing", "bars", "beats", "cut_notes", "file", "instruments", "instruments_progs", "is_quantized", "notes", "quantizer_args", "resolution", "subbeats", "subdivision_note", "tempo_changes", "time_signature_changes", "tonality", "total_bars"]
    class AbsoluteTiming(_message.Message):
        __slots__ = []
        def __init__(self) -> None: ...
    class Bar(_message.Message):
        __slots__ = ["absolute_timing", "bpm", "end_sec", "end_ticks", "harmonic_density", "note_density", "resolution", "start_sec", "start_ticks", "time_sig"]
        ABSOLUTE_TIMING_FIELD_NUMBER: _ClassVar[int]
        BPM_FIELD_NUMBER: _ClassVar[int]
        END_SEC_FIELD_NUMBER: _ClassVar[int]
        END_TICKS_FIELD_NUMBER: _ClassVar[int]
        HARMONIC_DENSITY_FIELD_NUMBER: _ClassVar[int]
        NOTE_DENSITY_FIELD_NUMBER: _ClassVar[int]
        RESOLUTION_FIELD_NUMBER: _ClassVar[int]
        START_SEC_FIELD_NUMBER: _ClassVar[int]
        START_TICKS_FIELD_NUMBER: _ClassVar[int]
        TIME_SIG_FIELD_NUMBER: _ClassVar[int]
        absolute_timing: bool
        bpm: float
        end_sec: float
        end_ticks: int
        harmonic_density: int
        note_density: int
        resolution: int
        start_sec: float
        start_ticks: int
        time_sig: str
        def __init__(self, bpm: _Optional[float] = ..., time_sig: _Optional[str] = ..., resolution: _Optional[int] = ..., absolute_timing: bool = ..., note_density: _Optional[int] = ..., harmonic_density: _Optional[int] = ..., start_ticks: _Optional[int] = ..., end_ticks: _Optional[int] = ..., start_sec: _Optional[float] = ..., end_sec: _Optional[float] = ...) -> None: ...
    class Beat(_message.Message):
        __slots__ = ["absolute_timing", "bar_idx", "bpm", "end_sec", "end_ticks", "global_idx", "resolution", "start_sec", "start_ticks", "time_sig"]
        ABSOLUTE_TIMING_FIELD_NUMBER: _ClassVar[int]
        BAR_IDX_FIELD_NUMBER: _ClassVar[int]
        BPM_FIELD_NUMBER: _ClassVar[int]
        END_SEC_FIELD_NUMBER: _ClassVar[int]
        END_TICKS_FIELD_NUMBER: _ClassVar[int]
        GLOBAL_IDX_FIELD_NUMBER: _ClassVar[int]
        RESOLUTION_FIELD_NUMBER: _ClassVar[int]
        START_SEC_FIELD_NUMBER: _ClassVar[int]
        START_TICKS_FIELD_NUMBER: _ClassVar[int]
        TIME_SIG_FIELD_NUMBER: _ClassVar[int]
        absolute_timing: bool
        bar_idx: int
        bpm: float
        end_sec: float
        end_ticks: int
        global_idx: int
        resolution: int
        start_sec: float
        start_ticks: int
        time_sig: str
        def __init__(self, bpm: _Optional[float] = ..., time_sig: _Optional[str] = ..., resolution: _Optional[int] = ..., absolute_timing: bool = ..., start_ticks: _Optional[int] = ..., end_ticks: _Optional[int] = ..., start_sec: _Optional[float] = ..., end_sec: _Optional[float] = ..., global_idx: _Optional[int] = ..., bar_idx: _Optional[int] = ...) -> None: ...
    class CutNotes(_message.Message):
        __slots__ = []
        def __init__(self) -> None: ...
    class File(_message.Message):
        __slots__ = []
        def __init__(self) -> None: ...
    class Instrument(_message.Message):
        __slots__ = ["family", "instrument", "is_drum", "name", "program"]
        FAMILY_FIELD_NUMBER: _ClassVar[int]
        INSTRUMENT_FIELD_NUMBER: _ClassVar[int]
        IS_DRUM_FIELD_NUMBER: _ClassVar[int]
        NAME_FIELD_NUMBER: _ClassVar[int]
        PROGRAM_FIELD_NUMBER: _ClassVar[int]
        family: str
        instrument: int
        is_drum: bool
        name: str
        program: int
        def __init__(self, instrument: _Optional[int] = ..., program: _Optional[int] = ..., name: _Optional[str] = ..., family: _Optional[str] = ..., is_drum: bool = ...) -> None: ...
    class InstrumentsProgs(_message.Message):
        __slots__ = []
        def __init__(self) -> None: ...
    class IsQuantized(_message.Message):
        __slots__ = []
        def __init__(self) -> None: ...
    class Note(_message.Message):
        __slots__ = ["bar_idx", "beat_idx", "end_sec", "end_ticks", "instrument_idx", "instrument_prog", "ligated", "note_name", "octave", "pitch", "pitch_name", "start_sec", "start_ticks", "subbeat_idx", "symbolic", "velocity"]
        BAR_IDX_FIELD_NUMBER: _ClassVar[int]
        BEAT_IDX_FIELD_NUMBER: _ClassVar[int]
        END_SEC_FIELD_NUMBER: _ClassVar[int]
        END_TICKS_FIELD_NUMBER: _ClassVar[int]
        INSTRUMENT_IDX_FIELD_NUMBER: _ClassVar[int]
        INSTRUMENT_PROG_FIELD_NUMBER: _ClassVar[int]
        LIGATED_FIELD_NUMBER: _ClassVar[int]
        NOTE_NAME_FIELD_NUMBER: _ClassVar[int]
        OCTAVE_FIELD_NUMBER: _ClassVar[int]
        PITCH_FIELD_NUMBER: _ClassVar[int]
        PITCH_NAME_FIELD_NUMBER: _ClassVar[int]
        START_SEC_FIELD_NUMBER: _ClassVar[int]
        START_TICKS_FIELD_NUMBER: _ClassVar[int]
        SUBBEAT_IDX_FIELD_NUMBER: _ClassVar[int]
        SYMBOLIC_FIELD_NUMBER: _ClassVar[int]
        VELOCITY_FIELD_NUMBER: _ClassVar[int]
        bar_idx: int
        beat_idx: int
        end_sec: float
        end_ticks: int
        instrument_idx: int
        instrument_prog: int
        ligated: bool
        note_name: str
        octave: str
        pitch: int
        pitch_name: str
        start_sec: float
        start_ticks: int
        subbeat_idx: int
        symbolic: str
        velocity: int
        def __init__(self, pitch: _Optional[int] = ..., pitch_name: _Optional[str] = ..., note_name: _Optional[str] = ..., octave: _Optional[str] = ..., ligated: bool = ..., start_ticks: _Optional[int] = ..., end_ticks: _Optional[int] = ..., start_sec: _Optional[float] = ..., end_sec: _Optional[float] = ..., symbolic: _Optional[str] = ..., velocity: _Optional[int] = ..., bar_idx: _Optional[int] = ..., beat_idx: _Optional[int] = ..., subbeat_idx: _Optional[int] = ..., instrument_idx: _Optional[int] = ..., instrument_prog: _Optional[int] = ...) -> None: ...
    class QuantizerArgs(_message.Message):
        __slots__ = []
        def __init__(self) -> None: ...
    class Resolution(_message.Message):
        __slots__ = []
        def __init__(self) -> None: ...
    class Subbeat(_message.Message):
        __slots__ = ["absolute_timing", "bar_idx", "beat_idx", "bpm", "end_sec", "end_ticks", "global_idx", "harmonic_density", "note_density", "resolution", "start_sec", "start_ticks", "time_sig"]
        ABSOLUTE_TIMING_FIELD_NUMBER: _ClassVar[int]
        BAR_IDX_FIELD_NUMBER: _ClassVar[int]
        BEAT_IDX_FIELD_NUMBER: _ClassVar[int]
        BPM_FIELD_NUMBER: _ClassVar[int]
        END_SEC_FIELD_NUMBER: _ClassVar[int]
        END_TICKS_FIELD_NUMBER: _ClassVar[int]
        GLOBAL_IDX_FIELD_NUMBER: _ClassVar[int]
        HARMONIC_DENSITY_FIELD_NUMBER: _ClassVar[int]
        NOTE_DENSITY_FIELD_NUMBER: _ClassVar[int]
        RESOLUTION_FIELD_NUMBER: _ClassVar[int]
        START_SEC_FIELD_NUMBER: _ClassVar[int]
        START_TICKS_FIELD_NUMBER: _ClassVar[int]
        TIME_SIG_FIELD_NUMBER: _ClassVar[int]
        absolute_timing: bool
        bar_idx: int
        beat_idx: int
        bpm: float
        end_sec: float
        end_ticks: int
        global_idx: int
        harmonic_density: int
        note_density: int
        resolution: int
        start_sec: float
        start_ticks: int
        time_sig: str
        def __init__(self, bpm: _Optional[float] = ..., time_sig: _Optional[str] = ..., resolution: _Optional[int] = ..., absolute_timing: bool = ..., note_density: _Optional[int] = ..., harmonic_density: _Optional[int] = ..., start_ticks: _Optional[int] = ..., end_ticks: _Optional[int] = ..., start_sec: _Optional[float] = ..., end_sec: _Optional[float] = ..., global_idx: _Optional[int] = ..., bar_idx: _Optional[int] = ..., beat_idx: _Optional[int] = ...) -> None: ...
    class SubdivisionNote(_message.Message):
        __slots__ = []
        def __init__(self) -> None: ...
    class TempoChanges(_message.Message):
        __slots__ = []
        def __init__(self) -> None: ...
    class TimeSignatureChanges(_message.Message):
        __slots__ = []
        def __init__(self) -> None: ...
    class Tonality(_message.Message):
        __slots__ = []
        def __init__(self) -> None: ...
    class TotalBars(_message.Message):
        __slots__ = []
        def __init__(self) -> None: ...
    ABSOLUTE_TIMING_FIELD_NUMBER: _ClassVar[int]
    BARS_FIELD_NUMBER: _ClassVar[int]
    BEATS_FIELD_NUMBER: _ClassVar[int]
    CUT_NOTES_FIELD_NUMBER: _ClassVar[int]
    FILE_FIELD_NUMBER: _ClassVar[int]
    INSTRUMENTS_FIELD_NUMBER: _ClassVar[int]
    INSTRUMENTS_PROGS_FIELD_NUMBER: _ClassVar[int]
    IS_QUANTIZED_FIELD_NUMBER: _ClassVar[int]
    NOTES_FIELD_NUMBER: _ClassVar[int]
    QUANTIZER_ARGS_FIELD_NUMBER: _ClassVar[int]
    RESOLUTION_FIELD_NUMBER: _ClassVar[int]
    SUBBEATS_FIELD_NUMBER: _ClassVar[int]
    SUBDIVISION_NOTE_FIELD_NUMBER: _ClassVar[int]
    TEMPO_CHANGES_FIELD_NUMBER: _ClassVar[int]
    TIME_SIGNATURE_CHANGES_FIELD_NUMBER: _ClassVar[int]
    TONALITY_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BARS_FIELD_NUMBER: _ClassVar[int]
    absolute_timing: _containers.RepeatedCompositeFieldContainer[Musa.AbsoluteTiming]
    bars: _containers.RepeatedCompositeFieldContainer[Musa.Bar]
    beats: _containers.RepeatedCompositeFieldContainer[Musa.Beat]
    cut_notes: _containers.RepeatedCompositeFieldContainer[Musa.CutNotes]
    file: _containers.RepeatedCompositeFieldContainer[Musa.File]
    instruments: _containers.RepeatedCompositeFieldContainer[Musa.Instrument]
    instruments_progs: _containers.RepeatedCompositeFieldContainer[Musa.InstrumentsProgs]
    is_quantized: _containers.RepeatedCompositeFieldContainer[Musa.IsQuantized]
    notes: _containers.RepeatedCompositeFieldContainer[Musa.Note]
    quantizer_args: _containers.RepeatedCompositeFieldContainer[Musa.QuantizerArgs]
    resolution: _containers.RepeatedCompositeFieldContainer[Musa.Resolution]
    subbeats: _containers.RepeatedCompositeFieldContainer[Musa.Subbeat]
    subdivision_note: _containers.RepeatedCompositeFieldContainer[Musa.SubdivisionNote]
    tempo_changes: _containers.RepeatedCompositeFieldContainer[Musa.TempoChanges]
    time_signature_changes: _containers.RepeatedCompositeFieldContainer[Musa.TimeSignatureChanges]
    tonality: _containers.RepeatedCompositeFieldContainer[Musa.Tonality]
    total_bars: _containers.RepeatedCompositeFieldContainer[Musa.TotalBars]
    def __init__(self, time_signature_changes: _Optional[_Iterable[_Union[Musa.TimeSignatureChanges, _Mapping]]] = ..., subdivision_note: _Optional[_Iterable[_Union[Musa.SubdivisionNote, _Mapping]]] = ..., file: _Optional[_Iterable[_Union[Musa.File, _Mapping]]] = ..., total_bars: _Optional[_Iterable[_Union[Musa.TotalBars, _Mapping]]] = ..., tonality: _Optional[_Iterable[_Union[Musa.Tonality, _Mapping]]] = ..., resolution: _Optional[_Iterable[_Union[Musa.Resolution, _Mapping]]] = ..., is_quantized: _Optional[_Iterable[_Union[Musa.IsQuantized, _Mapping]]] = ..., quantizer_args: _Optional[_Iterable[_Union[Musa.QuantizerArgs, _Mapping]]] = ..., absolute_timing: _Optional[_Iterable[_Union[Musa.AbsoluteTiming, _Mapping]]] = ..., cut_notes: _Optional[_Iterable[_Union[Musa.CutNotes, _Mapping]]] = ..., tempo_changes: _Optional[_Iterable[_Union[Musa.TempoChanges, _Mapping]]] = ..., instruments_progs: _Optional[_Iterable[_Union[Musa.InstrumentsProgs, _Mapping]]] = ..., instruments: _Optional[_Iterable[_Union[Musa.Instrument, _Mapping]]] = ..., bars: _Optional[_Iterable[_Union[Musa.Bar, _Mapping]]] = ..., notes: _Optional[_Iterable[_Union[Musa.Note, _Mapping]]] = ..., beats: _Optional[_Iterable[_Union[Musa.Beat, _Mapping]]] = ..., subbeats: _Optional[_Iterable[_Union[Musa.Subbeat, _Mapping]]] = ...) -> None: ...
