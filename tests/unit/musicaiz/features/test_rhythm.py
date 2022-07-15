import numpy as np


from musicaiz.structure import (
    Note,
)
from musicaiz.features import (
    get_ioi,
    get_start_sec,
    get_labeled_beat_vector,
    _delete_duplicates,
    _split_labeled_beat_vector,
    compute_rhythm_self_similarity_matrix,
)


def test_get_start_sec():
    notes = [
        Note(pitch=55, start=15.25, end=15.32, velocity=127),
        Note(pitch=79, start=16.75, end=16.78, velocity=127),
        Note(pitch=74, start=18.75, end=18.78, velocity=127),
        Note(pitch=55, start=18.77, end=18.79, velocity=127),
    ]

    expected = [15.25, 16.75, 18.75, 18.77]

    got = get_start_sec(notes)
    assert set(got) == set(expected)


def test_delete_duplicates():
    all_note_on = [0, 3, 4, 3, 3, 2, 2]
    expected = [0, 3, 4, 2]

    got = _delete_duplicates(all_note_on)
    assert set(got) == set(expected)


def test_get_ioi_a():

    all_note_on = [0.0, 1.0, 1.375]
    delete_overlap = True
    expected = [1.0, 0.375]

    got = get_ioi(all_note_on, delete_overlap)

    assert len(got) == len(expected)
    assert set(got) == set(expected)


def test_get_ioi_b():

    all_note_on = [0.0, 1.0, 1.0, 1.375]
    delete_overlap = False
    expected = [1.0, 0.0, 0.375]

    got = get_ioi(all_note_on, delete_overlap)

    assert len(got) == len(expected)
    assert set(got) == set(expected)


def test_get_labeled_beat_vector_a():
    # Test case: Paper example
    iois = [0.5, 0.375, 0.125]
    expected = [4, 4, 4, 4, 3, 3, 3, 1]

    got = get_labeled_beat_vector(iois)
    assert set(got) == set(expected)


def test_get_labeled_beat_vector_b():
    # Test case: Different IOI length
    iois = [1, 0.25]
    expected = [8, 8, 8, 8, 8, 8, 8, 8, 2, 2]

    got = get_labeled_beat_vector(iois)
    assert set(got) == set(expected)


def test_split_labeled_beat_vector_a():
    labeled_beat_vector = [8, 8, 8, 8, 8, 8, 8, 8, 2, 2]
    beat_value = 2

    expected = [
        [8, 8],
        [8, 8],
        [8, 8],
        [8, 8],
        [2, 2]
    ]

    got = _split_labeled_beat_vector(labeled_beat_vector, beat_value)
    for i in range(len(expected)):
        assert set(got[i]) == set(expected[i])


def test_split_labeled_beat_vector_b():
    labeled_beat_vector = [4, 4, 4, 4]
    beat_value = 4

    expected = [[4, 4, 4, 4]]

    got = _split_labeled_beat_vector(labeled_beat_vector, beat_value)
    for i in range(len(expected)):
        assert set(got[i]) == set(expected[i])


def test_split_labeled_beat_vector_c():
    # Test case: labeled beat vector length < beat value
    labeled_beat_vector = [1]
    beat_value = 12

    expected = [[1]]

    got = _split_labeled_beat_vector(labeled_beat_vector, beat_value)
    for i in range(len(expected)):
        assert got[i] == expected[i][0]


def test_compute_rhythm_self_similarity_matrix():
    splitted_beat_vector = [
        [8, 8],
        [8, 8],
        [8, 8],
        [8, 8],
        [2, 2]
    ]

    expected = np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0],
    ])

    got = compute_rhythm_self_similarity_matrix(splitted_beat_vector)
    comparison = got == expected
    assert comparison.all()




