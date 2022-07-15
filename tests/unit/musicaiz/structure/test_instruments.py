import pytest


# Our modules
from musicaiz.structure import (
    InstrumentMidiPrograms,
    InstrumentMidiFamilies,
    Instrument,
)


# ===============InstrumentMidiPrograms class Tests====
# =====================================================
def test_InstrumentMidiPrograms_get_possible_names_a():
    got = InstrumentMidiPrograms.ACOUSTIC_GRAND_PIANO.possible_names
    expected = [
        "ACOUSTIC_GRAND_PIANO",
        "ACOUSTIC GRAND PIANO",
        "acoustic_grand_piano",
        "acoustic grand piano"
    ]
    assert set(got) == set(expected)


def test_InstrumentMidiPrograms_get_all_instrument_names_b():
    got = InstrumentMidiPrograms.get_all_instrument_names()
    assert len(got) != 0


def test_InstrumentMidiPrograms_get_all_possible_names():
    got = InstrumentMidiPrograms.get_all_possible_names()
    assert len(got) != 0


def test_InstrumentMidiPrograms_check_name():
    name = "violin"
    got = InstrumentMidiPrograms._check_name(name)
    assert got


def test_InstrumentMidiPrograms_map_name():
    name = "acoustic grand piano"
    expected = InstrumentMidiPrograms.ACOUSTIC_GRAND_PIANO
    got = InstrumentMidiPrograms.map_name(name)
    assert expected == got


def test_InstrumentMidiPrograms_get_name_from_program():
    program = 1
    expected = InstrumentMidiPrograms.BRIGHT_ACOUSTIC_PIANO
    got = InstrumentMidiPrograms.get_name_from_program(program)
    assert expected == got


# ===============InstrumentMidiFamilies class Tests====
# =====================================================
def test_InstrumentMidiFamilies_get_family_from_instrument_name_a():
    # Test case: Non valid instrument name
    instrument_name = "piano"
    with pytest.raises(ValueError):
        InstrumentMidiFamilies.get_family_from_instrument_name(instrument_name)


def test_InstrumentMidiFamilies_get_family_from_instrument_name_b():
    # Test case: Valid instrument name
    instrument_name = "acoustic grand piano"
    expected = InstrumentMidiFamilies.PIANO
    got = InstrumentMidiFamilies.get_family_from_instrument_name(instrument_name)
    assert got == expected


# ===============Instrument class Tests================
# =====================================================
def test_Instrument_a():
    # Test case: Initializing with program and name
    program = 0
    name = "acoustic grand piano"
    instrument = Instrument(program, name)
    assert instrument.family == "PIANO"
    assert instrument.name == "ACOUSTIC_GRAND_PIANO"
    assert instrument.is_drum is False


def test_Instrument_b():
    # Test case: Initializing with program, name and custom is_drum
    program = 0
    name = "acoustic grand piano"
    is_drum = True
    instrument = Instrument(program, name, is_drum)
    assert instrument.family == "PIANO"
    assert instrument.name == "ACOUSTIC_GRAND_PIANO"
    assert instrument.is_drum is False


def test_Instrument_c():
    # Test case: Initializing with program and not name
    program = 3
    instrument = Instrument(program=program)
    assert instrument.family == "PIANO"
    assert instrument.name == "HONKY_TONK_PIANO"
    assert instrument.is_drum is False


def test_Instrument_d():
    # Test case: Initializing with valid name and not program
    name = "viola"
    instrument = Instrument(name=name)
    assert instrument.family == "STRINGS"
    assert instrument.name == "VIOLA"
    assert instrument.program == 41
    assert instrument.is_drum is False


# TODO: Add test where is_drum is true: Ex instr = "DRUMS_1"


def test_Instrument_e():
    # Test case: Initializing with invalid name and not program
    name = "yamaha piano"
    with pytest.raises(ValueError):
        Instrument(name=name)


def test_Instrument_f():
    # Test case: Initializing with no name nor program (wrong)
    with pytest.raises(ValueError):
        Instrument()
