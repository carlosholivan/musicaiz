from __future__ import annotations
import pretty_midi as pm
from enum import Enum
from typing import List
import numpy as np
from pathlib import Path

# try import fluidsynth
try:
    import fluidsynth
    _HAS_FLUIDSYNTH = True
except ImportError:
    _HAS_FLUIDSYNTH = False


class InstrumentMidiPrograms(Enum):
    # Value 1: List of Midi instrument program number
    ACOUSTIC_GRAND_PIANO = 0
    BRIGHT_ACOUSTIC_PIANO = 1
    ELECTRIC_GRAND_PIANO = 2
    HONKY_TONK_PIANO = 3
    ELECTRIC_PIANO_1 = 4
    ELECTRIC_PIANO_2 = 5
    HARPSICHORD = 6
    CLAVI = 7
    CELESTA = 8
    GLOCKENSPIEL = 9
    MUSIC_BOX = 10
    VIBRAPHONE = 11
    MARIMBA = 12
    XYLOPHONE = 13
    TUBULAR_BELLS = 14
    DULCIMER = 15
    DRAWBAR_ORGAN = 16
    PERCUSSIVE_ORGAN = 17
    ROCK_ORGAN = 18
    CHURCH_ORGAN = 19
    REED_ORGAN = 20
    ACCORDION = 21
    HARMONICA = 22
    TANGO_ACCORDION = 23
    ACOUSTIC_GUITAR_NYLON = 24
    ACOUSTIC_GUITAR_STEEL = 25
    ACOUSTIC_GUITAR_JAZZ = 26
    ACOUSTIC_GUITAR_CLEAN = 27
    ACOUSTIC_GUITAR_MUTED = 28
    OVERDRIVEN_GUITAR = 29
    DISTORSION_GUITAR = 30
    GUITAR_HARMONICS = 31
    ACOUSTIC_BASS = 32
    ELECTRIC_BASS_FINGER = 33
    ELECTRIC_BASS_PICK = 34
    FRETLESS_BASS = 35
    SLAP_BASS_1 = 36
    SLAP_BASS_2 = 37
    SYNTH_BASS_1 = 38
    SYNTH_BASS_2 = 39
    VIOLIN = 40
    VIOLA = 41
    CELLO = 42
    CONTRABASS = 43
    TREMOLO_STRINGS = 44
    PIZZICATO_STRINGS = 45
    ORCHESTRAL_HARP = 46
    TIMPANI = 47
    STRING_ENSEMBLE_1 = 48
    STRING_ENSEMBLE_2 = 49
    SYNTH_STRINGS_1 = 50
    SYNTH_STRINGS_2 = 51
    CHOIR_AAHS = 52
    VOICE_OOHS = 53
    SYNTH_VOICE = 54
    ORCHESTRA_HIT = 55
    TRUMPET = 56
    TROMBONE = 57
    TUBA = 58
    MUTED_TRUMPET = 59
    FRENCH_HORN = 60
    BRASS_SECTION = 61
    SYNTH_BRASS_1 = 62
    SYNTH_BRASS_2 = 63
    SOPRANO_SAX = 64
    ALTO_SAX = 65
    TENOR_SAX = 66
    BARITONE_SAX = 67
    OBOE = 68
    ENGLISH_HORN = 69
    BASOON = 70
    CLARINET = 71
    PICCOLO = 72
    FLUTE = 73
    RECORDER = 74
    PLAN_FLUTE = 75
    BLOWN_BOTTLE = 76
    SHAKUHACHI = 77
    WHISTLE = 78
    OCARINA = 79
    LEAD_1_SQUARE = 80
    LEAD_2_SAWTOOTH = 81
    LEAD_3_CALLIOPE = 82
    LEAD_4_CHIFF = 83
    LEAD_5_CHARANG = 84
    LEAD_6_VOICE = 85
    LEAD_7_FIFTHS = 86
    LEAD_8_BASS_LEAD = 87
    PAD_1_NEW_AGE = 88
    PAD_2_WARM = 89
    PAD_3_POLYSYNTH = 90
    PAD_4_CHOIR = 91
    PAD_5_COWED = 92
    PAD_6_METALLIC = 93
    PAD_7_HALO = 94
    PAD_8_SWEEP = 95
    FX_1_RAIN = 96
    FX_2_SOUNDTRACK = 97
    FX_3_CRYSTAL = 98
    FX_4_ATMOSPHERE = 99
    FX_5_BRIGHTNESS = 100
    FX_6_GOBLINS = 101
    FX_7_ECHOES = 102
    FX_8_SCI_FI = 103
    SITAR = 104
    BANJO = 105
    SHAMISEN = 106
    KOTO = 107
    KALIMBA = 108
    BAG_PIPE = 109
    FIDDLE = 110
    SHANAI = 111
    TINKLE_BELL = 112
    AGOGO = 113
    STEEL_DRUMS = 114
    WOODBLOCK = 115
    TAIKO_DRUM = 116
    MELODIC_TOM = 117
    SYNTH_DRUM = 118
    REVERSE_CYMBAL = 119
    GUITAR_FRET_NOISE = 120
    BREADTH_NOISE = 121
    SEASHORE = 122
    BIRD_TWEET = 123
    TELEPHONE_RING = 124
    HELICOPTER = 125
    APPLAUSE = 126
    GUNSHOT = 127

    @property
    def possible_names(self) -> List[str]:
        instr_naming = []
        instr_naming += [self.name.lower()]
        instr_naming += [self.name]
        if "_" in self.name:
            instr_naming += [self.name.replace("_", " ")]
            instr_naming += [self.name.lower().replace("_", " ")]
        return instr_naming

    @classmethod
    def get_all_instrument_names(cls) -> List[str]:
        return [name for name in cls.__members__]

    @classmethod
    def get_all_possible_names(cls) -> List[str]:
        all_names = []
        for instr in cls.__members__.values():
            all_names.extend(instr.possible_names)
        return all_names

    @classmethod
    def map_name(cls, name: str) -> InstrumentMidiPrograms:
        """Get the instrument enum name from a possible name (lower typing...)."""
        if not cls._check_name(name):
            raise ValueError(f"Instrument name {name} not found.")
        if " " in name:
            name = name.replace(" ", "_")
        return cls[name.upper()]

    @classmethod
    def _check_name(cls, name: str) -> bool:
        if name not in cls.get_all_possible_names():
            return False
        else:
            return True

    @classmethod
    def get_name_from_program(cls, program: int) -> InstrumentMidiPrograms:
        prog = None
        for i in cls.__members__.values():
            if i.value == program:
                prog = i
        if prog is None:
            raise ValueError(f"Program {program} is not valid.")
        return prog


# TODO: Doubt: Is this better to be a dict so we can introduce
# directly the instrument name and program n to not have so many lines for this?
class InstrumentMidiFamilies(Enum):
    """
    Value 1: List of Midi instrument program number
    Value 2: Color (for plot purposes)
    Value 3: Pitch range?

    MIDI Instrument families:
        - Piano 1-8
        - Chromatic Percussion 9-16
        - Organ 17-24
        - Guitar 25-32
        - Bass 33-40
        - Strings 41-48
        - Ensemble 49-56
        - Brass 57-64
        - Reed 65-72
        - Pipe 73-80
        - Synth Lead 81-88
        - Synth Pad 89-96
        - Synth Effects 97-104
        - Ethnic 105-112
        - Percussion 113-119
        - Sound Effects 120-127
    """
    PIANO = [
        InstrumentMidiPrograms.ACOUSTIC_GRAND_PIANO,
        InstrumentMidiPrograms.BRIGHT_ACOUSTIC_PIANO,
        InstrumentMidiPrograms.ELECTRIC_GRAND_PIANO,
        InstrumentMidiPrograms.HONKY_TONK_PIANO,
        InstrumentMidiPrograms.ELECTRIC_PIANO_1,
        InstrumentMidiPrograms.ELECTRIC_PIANO_2,
        InstrumentMidiPrograms.HARPSICHORD,
        InstrumentMidiPrograms.CLAVI,
        InstrumentMidiPrograms.CELESTA
    ]
    CHROMATIC_PERCUSSION = [
        InstrumentMidiPrograms.CELESTA,
        InstrumentMidiPrograms.GLOCKENSPIEL,
        InstrumentMidiPrograms.MUSIC_BOX,
        InstrumentMidiPrograms.VIBRAPHONE,
        InstrumentMidiPrograms.MARIMBA,
        InstrumentMidiPrograms.XYLOPHONE,
        InstrumentMidiPrograms.TUBULAR_BELLS,
        InstrumentMidiPrograms.DULCIMER,
    ]
    ORGAN = [
        InstrumentMidiPrograms.DRAWBAR_ORGAN,
        InstrumentMidiPrograms.PERCUSSIVE_ORGAN,
        InstrumentMidiPrograms.ROCK_ORGAN,
        InstrumentMidiPrograms.CHURCH_ORGAN,
        InstrumentMidiPrograms.REED_ORGAN,
        InstrumentMidiPrograms.ACCORDION,
        InstrumentMidiPrograms.HARMONICA,
        InstrumentMidiPrograms.TANGO_ACCORDION,
    ]
    GUITAR = [
        InstrumentMidiPrograms.ACOUSTIC_GUITAR_NYLON,
        InstrumentMidiPrograms.ACOUSTIC_GUITAR_STEEL,
        InstrumentMidiPrograms.ACOUSTIC_GUITAR_JAZZ,
        InstrumentMidiPrograms.ACOUSTIC_GUITAR_CLEAN,
        InstrumentMidiPrograms.ACOUSTIC_GUITAR_MUTED,
        InstrumentMidiPrograms.OVERDRIVEN_GUITAR,
        InstrumentMidiPrograms.DISTORSION_GUITAR,
        InstrumentMidiPrograms.GUITAR_HARMONICS,
    ]
    BASS = [
        InstrumentMidiPrograms.ACOUSTIC_BASS,
        InstrumentMidiPrograms.ELECTRIC_BASS_FINGER,
        InstrumentMidiPrograms.ELECTRIC_BASS_PICK,
        InstrumentMidiPrograms.FRETLESS_BASS,
        InstrumentMidiPrograms.SLAP_BASS_1,
        InstrumentMidiPrograms.SLAP_BASS_2,
        InstrumentMidiPrograms.SYNTH_BASS_1,
        InstrumentMidiPrograms.SYNTH_BASS_2,
    ]
    STRINGS = [
        InstrumentMidiPrograms.VIOLIN,
        InstrumentMidiPrograms.VIOLA,
        InstrumentMidiPrograms.CELLO,
        InstrumentMidiPrograms.CONTRABASS,
        InstrumentMidiPrograms.TREMOLO_STRINGS,
        InstrumentMidiPrograms.PIZZICATO_STRINGS,
        InstrumentMidiPrograms.ORCHESTRAL_HARP,
        InstrumentMidiPrograms.TIMPANI,
    ]
    ENSEMBLE = [
        InstrumentMidiPrograms.STRING_ENSEMBLE_1,
        InstrumentMidiPrograms.STRING_ENSEMBLE_2,
        InstrumentMidiPrograms.SYNTH_STRINGS_1,
        InstrumentMidiPrograms.SYNTH_STRINGS_2,
        InstrumentMidiPrograms.CHOIR_AAHS,
        InstrumentMidiPrograms.VOICE_OOHS,
        InstrumentMidiPrograms.SYNTH_VOICE,
        InstrumentMidiPrograms.ORCHESTRA_HIT,
    ]
    BRASS = [
        InstrumentMidiPrograms.TRUMPET,
        InstrumentMidiPrograms.TROMBONE,
        InstrumentMidiPrograms.TUBA,
        InstrumentMidiPrograms.MUTED_TRUMPET,
        InstrumentMidiPrograms.FRENCH_HORN,
        InstrumentMidiPrograms.BRASS_SECTION,
        InstrumentMidiPrograms.SYNTH_BRASS_1,
        InstrumentMidiPrograms.SYNTH_BRASS_2,
    ]
    REED = [
        InstrumentMidiPrograms.SOPRANO_SAX,
        InstrumentMidiPrograms.ALTO_SAX,
        InstrumentMidiPrograms.TENOR_SAX,
        InstrumentMidiPrograms.BARITONE_SAX,
        InstrumentMidiPrograms.OBOE,
        InstrumentMidiPrograms.ENGLISH_HORN,
        InstrumentMidiPrograms.BASOON,
        InstrumentMidiPrograms.CLARINET,
    ]
    PIPE = [
        InstrumentMidiPrograms.PICCOLO,
        InstrumentMidiPrograms.FLUTE,
        InstrumentMidiPrograms.RECORDER,
        InstrumentMidiPrograms.PLAN_FLUTE,
        InstrumentMidiPrograms.BLOWN_BOTTLE,
        InstrumentMidiPrograms.SHAKUHACHI,
        InstrumentMidiPrograms.WHISTLE,
        InstrumentMidiPrograms.OCARINA,
    ]
    SYNTH_LEAD = [
        InstrumentMidiPrograms.LEAD_1_SQUARE,
        InstrumentMidiPrograms.LEAD_2_SAWTOOTH,
        InstrumentMidiPrograms.LEAD_3_CALLIOPE,
        InstrumentMidiPrograms.LEAD_4_CHIFF,
        InstrumentMidiPrograms.LEAD_5_CHARANG,
        InstrumentMidiPrograms.LEAD_6_VOICE,
        InstrumentMidiPrograms.LEAD_7_FIFTHS,
        InstrumentMidiPrograms.LEAD_8_BASS_LEAD,
    ]
    SYNTH_PAD = [
        InstrumentMidiPrograms.PAD_1_NEW_AGE,
        InstrumentMidiPrograms.PAD_2_WARM,
        InstrumentMidiPrograms.PAD_3_POLYSYNTH,
        InstrumentMidiPrograms.PAD_4_CHOIR,
        InstrumentMidiPrograms.PAD_5_COWED,
        InstrumentMidiPrograms.PAD_6_METALLIC,
        InstrumentMidiPrograms.PAD_7_HALO,
        InstrumentMidiPrograms.PAD_8_SWEEP,
    ]
    SYNTH_EFFECTS = [
        InstrumentMidiPrograms.FX_1_RAIN,
        InstrumentMidiPrograms.FX_2_SOUNDTRACK,
        InstrumentMidiPrograms.FX_3_CRYSTAL,
        InstrumentMidiPrograms.FX_4_ATMOSPHERE,
        InstrumentMidiPrograms.FX_5_BRIGHTNESS,
        InstrumentMidiPrograms.FX_6_GOBLINS,
        InstrumentMidiPrograms.FX_7_ECHOES,
        InstrumentMidiPrograms.FX_8_SCI_FI,
    ]
    ETHNIC = [
        InstrumentMidiPrograms.SITAR,
        InstrumentMidiPrograms.BANJO,
        InstrumentMidiPrograms.SHAMISEN,
        InstrumentMidiPrograms.KOTO,
        InstrumentMidiPrograms.KALIMBA,
        InstrumentMidiPrograms.BAG_PIPE,
        InstrumentMidiPrograms.FIDDLE,
        InstrumentMidiPrograms.SHANAI,
    ]
    PERCUSSION = [
        InstrumentMidiPrograms.SHANAI,
        InstrumentMidiPrograms.TINKLE_BELL,
        InstrumentMidiPrograms.AGOGO,
        InstrumentMidiPrograms.STEEL_DRUMS,
        InstrumentMidiPrograms.WOODBLOCK,
        InstrumentMidiPrograms.TAIKO_DRUM,
        InstrumentMidiPrograms.MELODIC_TOM,
        InstrumentMidiPrograms.SYNTH_DRUM,
        InstrumentMidiPrograms.REVERSE_CYMBAL,
    ]
    SOUND_EFFECTS = [
        InstrumentMidiPrograms.GUITAR_FRET_NOISE,
        InstrumentMidiPrograms.BREADTH_NOISE,
        InstrumentMidiPrograms.SEASHORE,
        InstrumentMidiPrograms.BIRD_TWEET,
        InstrumentMidiPrograms.TELEPHONE_RING,
        InstrumentMidiPrograms.HELICOPTER,
        InstrumentMidiPrograms.APPLAUSE,
        InstrumentMidiPrograms.GUNSHOT,
    ]

    @property
    def program_range(self):
        """Returns the list of instrument programs for an instrument family"""
        # InstrumentMidiFamilies(self.value).value[0]
        pass

    @classmethod
    def get_family_from_instrument_name(cls, instr_name: str) -> InstrumentMidiFamilies:
        family = None
        for i in cls.__members__.values():
            for instr in i.value:
                # Obtain all possible naming typing for an instrument
                if instr_name in instr.possible_names:
                    family = i
        if family is None:
            raise ValueError(f"Input instrument {instr_name} is not a valid instrument.")
        return family

    def get_family_from_instrument_program(self):
        pass


# TODO: Add channel as attribute?
# TODO: Add soundfont?
# TODO: Add bank, preset...?
class Instrument:

    def __init__(
        self,
        program: int = None,
        name: str = None,
        family: str = None,
        is_drum: bool = None,
        general_midi: bool = True,
    ):
        """
        - Instrument name or/and program must be provided.
        - If name is pprovided and no program is provided, the program will
        be automatically set with the MIDI instrument program according to
        the instrument name.
        - If program is provided but no instrument name is provided, the instrument
        name will be automatically set with the MIDI instrument name according to
        the instrument program number.
        - If both instrument name and program number are provided, both values will be set
        as the input arguments (this is useful in case of personal definition of MIDI instrument programs)
        - If general_midi is False, then the program number and name provided will
        set the program and name attributes nad we won't map the program to the instrument
        name nor viceversa according to general midi especification.
        - If general midi is False, we can also provide family if we know it, otherwise it'll be set to none.
        """

        # If user has a program number and instrument name mapping different to general midi
        # program number and name must be provided as they cannot be matched automatically.
        if not general_midi:
            if program is None or name is None:
                raise ValueError("Program number and instrument name must be provided.")
            self.program = program
            self.name = name
            self.family = family
        else:
            if program is not None:
                self.program = program
                if name is None or name == "":
                    name_obj = InstrumentMidiPrograms.get_name_from_program(program)
                else:
                    if general_midi:
                        name_obj = InstrumentMidiPrograms.map_name(name)
                self.name = name_obj.name
            elif name is not None:
                name_obj = InstrumentMidiPrograms.map_name(name)
                self.name = name_obj.name
                if program is None:
                    # If no program is provided we have to map the instr name to the program.
                    # To map the program with the input name, the name must be a MIDI instr name
                    # this checker is written in `InstrumentMidiPrograms._check_name` method.
                    self.program = name_obj.value
                else:
                    self.program = program
            elif name is None and program is None:
                raise ValueError("Instrument name or program must be provided.")
            family = InstrumentMidiFamilies.get_family_from_instrument_name(self.name)
            self.family = family.name

        # If user want to define their own instr definition they can set manually if the
        # instr is drum or not. If this var is not set up, it'll be initialized according to
        # MIDI instr definitions.
        if is_drum is None:
            if self.program > 127:
                self.is_drum = True
            else:
                self.is_drum = False
        else:
            self.is_drum = is_drum

        # List of notes in the instrument
        self.notes = []

        # List of bars in the instrument
        self.bars = []

    def __repr__(self):
        if self.family is None:
            family = "unknown"
        else:
            family = self.family
        return "Instrument(program={}, is_drum={}, name='{}', family={})".format(
            self.program,
            self.is_drum,
            self.name.replace('"', r'\"'),
            family.lower().replace('"', r'\"'),
        )
