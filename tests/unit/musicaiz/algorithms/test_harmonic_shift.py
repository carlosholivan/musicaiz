from musicaiz.structure import Note
from musicaiz.algorithms import scale_change

def test_scale_change_a():
    # Test case: All the origin_notes belong to the origin_tonality and scale
    origin_notes = [
        Note(pitch=43, start=0, end=96, velocity=82), # G
        Note(pitch=59, start=96, end=96*2, velocity=82), # B
        Note(pitch=60, start=96*2, end=96*3, velocity=82), # C
        Note(pitch=72, start=96*4, end=96*5, velocity=44) # C
    ]
    origin_tonality = "C_MAJOR"
    origin_scale = "MAJOR"
    target_tonality = "G_MAJOR"
    target_scale = "MAJOR"
    correction = True

    expected_notes = [
        Note(pitch=38, start=0, end=96, velocity=82), # D
        Note(pitch=54, start=96, end=96*2, velocity=82), # F#
        Note(pitch=67, start=96*2, end=96*3, velocity=82), # G
        Note(pitch=79, start=96*4, end=96*5, velocity=44) # G
    ]

    got_notes = scale_change(
        origin_notes,
        origin_tonality,
        origin_scale,
        target_tonality,
        target_scale,
        correction
    )

    for i, note in enumerate(expected_notes):
        assert expected_notes[i].pitch == got_notes[i].pitch


def test_scale_change_b():
    # Test case: Some origin_notes do not belong to the origin_torality and scale.
    # Correction applied.
    origin_notes = [
        Note(pitch=44, start=0, end=96, velocity=82), # G#
    ]
    origin_tonality = "C_MAJOR"
    origin_scale = "MAJOR"
    target_tonality = "G_MAJOR"
    target_scale = "MAJOR"
    correction = True

    expected_notes = [
        Note(pitch=40, start=0, end=96, velocity=82), # E
    ]

    got_notes = scale_change(
        origin_notes,
        origin_tonality,
        origin_scale,
        target_tonality,
        target_scale,
        correction
    )

    for i, note in enumerate(expected_notes):
        assert expected_notes[i].pitch == got_notes[i].pitch


def test_scale_change_c():
    # Test case: Some origin_notes do not belong to the origin_torality and scale.
    # No correction applied.
    origin_notes = [
        Note(pitch=44, start=0, end=96, velocity=82), # G#
    ]
    origin_tonality = "C_MAJOR"
    origin_scale = "MAJOR"
    target_tonality = "G_MAJOR"
    target_scale = "MAJOR"
    correction = False

    expected_notes = [
        Note(pitch=39, start=0, end=96, velocity=82), # D#
    ]

    got_notes = scale_change(
        origin_notes,
        origin_tonality,
        origin_scale,
        target_tonality,
        target_scale,
        correction
    )

    for i, note in enumerate(expected_notes):
        assert expected_notes[i].pitch == got_notes[i].pitch
