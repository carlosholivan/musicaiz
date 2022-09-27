from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Musa(_message.Message):
    __slots__ = ["instruments", "time_signatures"]
    class Instrument(_message.Message):
        __slots__ = ["bars", "family", "instrument", "is_drum", "name", "notes", "program"]
        class Bar(_message.Message):
            __slots__ = ["absolute_timing", "bpm", "end_sec", "end_ticks", "harmonic_density", "note_density", "notes", "resolution", "start_sec", "start_ticks", "time_sig"]
            class Note(_message.Message):
                __slots__ = ["end_sec", "end_ticks", "ligated", "note_name", "octave", "pitch", "pitch_name", "start_sec", "start_ticks", "symbolic", "velocity"]
                END_SEC_FIELD_NUMBER: _ClassVar[int]
                END_TICKS_FIELD_NUMBER: _ClassVar[int]
                LIGATED_FIELD_NUMBER: _ClassVar[int]
                NOTE_NAME_FIELD_NUMBER: _ClassVar[int]
                OCTAVE_FIELD_NUMBER: _ClassVar[int]
                PITCH_FIELD_NUMBER: _ClassVar[int]
                PITCH_NAME_FIELD_NUMBER: _ClassVar[int]
                START_SEC_FIELD_NUMBER: _ClassVar[int]
                START_TICKS_FIELD_NUMBER: _ClassVar[int]
                SYMBOLIC_FIELD_NUMBER: _ClassVar[int]
                VELOCITY_FIELD_NUMBER: _ClassVar[int]
                end_sec: float
                end_ticks: int
                ligated: bool
                note_name: str
                octave: str
                pitch: int
                pitch_name: str
                start_sec: float
                start_ticks: int
                symbolic: str
                velocity: int
                def __init__(self, pitch: _Optional[int] = ..., pitch_name: _Optional[str] = ..., note_name: _Optional[str] = ..., octave: _Optional[str] = ..., ligated: bool = ..., start_ticks: _Optional[int] = ..., end_ticks: _Optional[int] = ..., start_sec: _Optional[float] = ..., end_sec: _Optional[float] = ..., symbolic: _Optional[str] = ..., velocity: _Optional[int] = ...) -> None: ...
            ABSOLUTE_TIMING_FIELD_NUMBER: _ClassVar[int]
            BPM_FIELD_NUMBER: _ClassVar[int]
            END_SEC_FIELD_NUMBER: _ClassVar[int]
            END_TICKS_FIELD_NUMBER: _ClassVar[int]
            HARMONIC_DENSITY_FIELD_NUMBER: _ClassVar[int]
            NOTES_FIELD_NUMBER: _ClassVar[int]
            NOTE_DENSITY_FIELD_NUMBER: _ClassVar[int]
            RESOLUTION_FIELD_NUMBER: _ClassVar[int]
            START_SEC_FIELD_NUMBER: _ClassVar[int]
            START_TICKS_FIELD_NUMBER: _ClassVar[int]
            TIME_SIG_FIELD_NUMBER: _ClassVar[int]
            absolute_timing: bool
            bpm: int
            end_sec: float
            end_ticks: int
            harmonic_density: int
            note_density: int
            notes: _containers.RepeatedCompositeFieldContainer[Musa.Instrument.Bar.Note]
            resolution: int
            start_sec: float
            start_ticks: int
            time_sig: str
            def __init__(self, bpm: _Optional[int] = ..., time_sig: _Optional[str] = ..., resolution: _Optional[int] = ..., absolute_timing: bool = ..., note_density: _Optional[int] = ..., harmonic_density: _Optional[int] = ..., start_ticks: _Optional[int] = ..., end_ticks: _Optional[int] = ..., start_sec: _Optional[float] = ..., end_sec: _Optional[float] = ..., notes: _Optional[_Iterable[_Union[Musa.Instrument.Bar.Note, _Mapping]]] = ...) -> None: ...
        class Note(_message.Message):
            __slots__ = ["end_sec", "end_ticks", "ligated", "note_name", "octave", "pitch", "pitch_name", "start_sec", "start_ticks", "symbolic", "velocity"]
            END_SEC_FIELD_NUMBER: _ClassVar[int]
            END_TICKS_FIELD_NUMBER: _ClassVar[int]
            LIGATED_FIELD_NUMBER: _ClassVar[int]
            NOTE_NAME_FIELD_NUMBER: _ClassVar[int]
            OCTAVE_FIELD_NUMBER: _ClassVar[int]
            PITCH_FIELD_NUMBER: _ClassVar[int]
            PITCH_NAME_FIELD_NUMBER: _ClassVar[int]
            START_SEC_FIELD_NUMBER: _ClassVar[int]
            START_TICKS_FIELD_NUMBER: _ClassVar[int]
            SYMBOLIC_FIELD_NUMBER: _ClassVar[int]
            VELOCITY_FIELD_NUMBER: _ClassVar[int]
            end_sec: float
            end_ticks: int
            ligated: bool
            note_name: str
            octave: str
            pitch: int
            pitch_name: str
            start_sec: float
            start_ticks: int
            symbolic: str
            velocity: int
            def __init__(self, pitch: _Optional[int] = ..., pitch_name: _Optional[str] = ..., note_name: _Optional[str] = ..., octave: _Optional[str] = ..., ligated: bool = ..., start_ticks: _Optional[int] = ..., end_ticks: _Optional[int] = ..., start_sec: _Optional[float] = ..., end_sec: _Optional[float] = ..., symbolic: _Optional[str] = ..., velocity: _Optional[int] = ...) -> None: ...
        BARS_FIELD_NUMBER: _ClassVar[int]
        FAMILY_FIELD_NUMBER: _ClassVar[int]
        INSTRUMENT_FIELD_NUMBER: _ClassVar[int]
        IS_DRUM_FIELD_NUMBER: _ClassVar[int]
        NAME_FIELD_NUMBER: _ClassVar[int]
        NOTES_FIELD_NUMBER: _ClassVar[int]
        PROGRAM_FIELD_NUMBER: _ClassVar[int]
        bars: _containers.RepeatedCompositeFieldContainer[Musa.Instrument.Bar]
        family: str
        instrument: int
        is_drum: bool
        name: str
        notes: _containers.RepeatedCompositeFieldContainer[Musa.Instrument.Note]
        program: int
        def __init__(self, instrument: _Optional[int] = ..., program: _Optional[int] = ..., name: _Optional[str] = ..., family: _Optional[str] = ..., is_drum: bool = ..., bars: _Optional[_Iterable[_Union[Musa.Instrument.Bar, _Mapping]]] = ..., notes: _Optional[_Iterable[_Union[Musa.Instrument.Note, _Mapping]]] = ...) -> None: ...
    class Note(_message.Message):
        __slots__ = []
        def __init__(self) -> None: ...
    class TimeSignature(_message.Message):
        __slots__ = ["denom", "num"]
        DENOM_FIELD_NUMBER: _ClassVar[int]
        NUM_FIELD_NUMBER: _ClassVar[int]
        denom: int
        num: int
        def __init__(self, num: _Optional[int] = ..., denom: _Optional[int] = ...) -> None: ...
    class Tonality(_message.Message):
        __slots__ = []
        def __init__(self) -> None: ...
    INSTRUMENTS_FIELD_NUMBER: _ClassVar[int]
    TIME_SIGNATURES_FIELD_NUMBER: _ClassVar[int]
    instruments: _containers.RepeatedCompositeFieldContainer[Musa.Instrument]
    time_signatures: _containers.RepeatedCompositeFieldContainer[Musa.TimeSignature]
    def __init__(self, time_signatures: _Optional[_Iterable[_Union[Musa.TimeSignature, _Mapping]]] = ..., instruments: _Optional[_Iterable[_Union[Musa.Instrument, _Mapping]]] = ...) -> None: ...
