"""
Algorithms
==========

This module allows to extract, modify and predict different aspects of a music piece.


Harmonic Shifting
-----------------

.. autosummary::
    :toctree: generated/

    harmonic_shifting

Key-Profiles
------------

State of the Art algorithms for key finding in symbolic music.
The key-profiles weights in the library are the following:

- KRUMHANSL_KESSLER

[1] Krumhansl, C. L., & Kessler, E. J. (1982).
Tracing the dynamic changes in perceived tonal organization in a spatial representation of musical keys.
Psychological review, 89(4), 334.

- TEMPERLEY

[2] Temperley, D. (1999).
What's key for key? The Krumhansl-Schmuckler key-finding algorithm reconsidered.
Music Perception, 17(1), 65-100.

- ALBRETCH_SHANAHAN

[3] Albrecht, J., & Shanahan, D. (2013).
The use of large corpora to train a new type of key-finding algorithm: An improved treatment of the minor mode.
Music Perception: An Interdisciplinary Journal, 31(1), 59-67.

- SIGNATURE_FIFTHS


.. autosummary::
    :toctree: generated/

    KeyDetectionAlgorithms
    KrumhanslKessler
    Temperley
    AlbrechtShanahan
    key_detection

Chord Prediction
----------------

Predict chords at beat-level.

.. autosummary::
    :toctree: generated/

    predict_chords
    get_chords
    get_chords_candidates
    compute_chord_notes_dist
    _notes_to_onehot
"""

from .harmonic_shift import (
    harmonic_shifting,
    scale_change
)

from .key_profiles import (
    KeyDetectionAlgorithms,
    KrumhanslKessler,
    Temperley,
    AlbrechtShanahan,
    signature_fifths,
    _signature_fifths_keys,
    _right_left_notes,
    _correlation,
    _keys_correlations,
    signature_fifths_profiles,
    _eights_per_pitch_class,
    key_detection
)

from .chord_prediction import (
    predict_chords,
    get_chords,
    get_chords_candidates,
    compute_chord_notes_dist,
    _notes_to_onehot,
)


__all__ = [
    "harmonic_shifting",
    "scale_change",
    "KeyDetectionAlgorithms",
    "KrumhanslKessler",
    "Temperley",
    "AlbrechtShanahan",
    "signature_fifths",
    "_signature_fifths_keys",
    "_right_left_notes",
    "_correlation",
    "_keys_correlations",
    "signature_fifths_profiles",
    "_eights_per_pitch_class",
    "key_detection",
    "predict_chords",
    "get_chords",
    "get_chords_candidates",
    "compute_chord_notes_dist",
    "_notes_to_onehot",
]
