"""
Features
========

This module provides methods that allows to analyze symbolic music.

Pitch
-----

.. autosummary::
    :toctree: generated/

    get_highest_lowest_pitches
    get_pitch_range
    get_note_density
    get_pitch_classes
    get_note_classes
    pitch_counts
    get_last_note_class
    pitch_class_histogram
    pitch_class_transition_matrix
    plot_pitch_class_transition_matrix
    average_pitch_interval


Harmony
-------

.. autosummary::
    :toctree: generated/

    get_chord_type_from_note_seq
    get_intervals_note_seq
    predict_chords
    predict_scales_degrees
    predict_possible_progressions
    predict_progression
    _all_note_seq_permutations
    _delete_repeated_note_names
    _extract_note_positions
    _order_note_seq_by_chromatic_idx
    get_harmonic_density


Rhythm
------
This submodule contains the implementation of part of the paper:

.. panels::

    [1] Roig, C., Tardón, L. J., Barbancho, I., & Barbancho, A. M. (2014).
    Automatic melody composition based on a probabilistic model of music
    style and harmonic rules. Knowledge-Based Systems, 71, 419-434.
    http://dx.doi.org/10.1016/j.knosys.2014.08.018


The implementation follows the paper method to predict rhythmic patterns.
This module contains:

    1. Tempo (or bpm) estimation
        - get IOIs with `get_ioi` method.
        - get error ej

    2. Time signature estimation
        - get the labeled beat vector (or IOI') from IOIs with `get_labeled_beat_vector`
        - get the Bar Split Vectors (BSV) for each beat (k in the paper) with `get_split_bar_vector`.
        k goes from 2 to 12 which are the most-common time_sig numerators.
        - compute the RSSM with each BSV with `compute_rhythm_self_similarity_matrix`.
        - get the time_sig numerator which will be the RSSM with the highest repeated bar instances.

    3. Rhythm extraction

    4. Pitch contour extraction

.. autosummary::
    :toctree: generated/

    get_start_sec
    get_ioi
    get_labeled_beat_vector
    compute_rhythm_self_similarity_matrix
    plot_rmss
    compute_all_rmss
    get_symbolic_length_classes
    note_length_histogram
    note_length_transition_matrix
    plot_note_length_transition_matrix


Self-Similarity Matrices
------------------------

This submodule presents different implementations of self-similarity matrices.

The papers that are implemented in this sumbodule are the following:

.. panels::

    [1] Louie, W.
    MusicPlot: Interactive Self-Similarity Matrix for Music Structure Visualization.
    https://wlouie1.github.io/MusicPlot/musicplot_paper.pdf


The process to obtain the SSM with this method is:
1. Group the notes in bars and subdivisions.
2. Extract the highest note in each subdivision.
3. Calculate m_prime = [p1-p2, d2/d1, ...] with p the pitch and d the note duration.
4. Compute the SSM function.

.. autosummary::
    :toctree: generated/

    compute_ssm
    self_similarity_louie
    self_similarity_single_measure
    self_similarity_measures
    plot_ssm
    _self_similarity
    binarize_self_similarity_matrix
    feature_vector
    get_novelty_func
    get_segment_boundaries
    plot_novelty_from_ssm


Graphs
------

This submodule presents different implementations of self-similarity matrices.

The papers that are implemented in this sumbodule are the following:

.. panels::

    [1] Jeong, D., Kwon, T., Kim, Y., & Nam, J. (2019)
    Graph neural network for music score data and modeling expressive piano performance.
    In International Conference on Machine Learning, 3060-3070
    https://proceedings.mlr.press/v97/jeong19a.html


.. autosummary::
    :toctree: generated/

    musa_to_graph
    plot_graph


Form or Structure
-----------------

.. autosummary::
    :toctree: generated/

    PeltArgs
    StructurePrediction

"""


from .pitch import (
    get_highest_lowest_pitches,
    get_pitch_range,
    get_pitch_classes,
    get_note_classes,
    get_note_density,
    get_note_classes,
    get_last_note_class,
    pitch_class_histogram,
    pitch_class_transition_matrix,
    plot_pitch_class_transition_matrix,
    pitch_counts,
    average_pitch_interval,
)
from .harmony import (
    get_chord_type_from_note_seq,
    get_intervals_note_seq,
    predict_chords,
    predict_scales_degrees,
    predict_possible_progressions,
    predict_progression,
    get_harmonic_density,
    _all_note_seq_permutations,
    _delete_repeated_note_names,
    _extract_note_positions,
    _order_note_seq_by_chromatic_idx,
)
from .predict_midi import (
    predic_time_sig_numerator,
)
from .rhythm import (
    get_start_sec,
    get_ioi,
    _delete_duplicates,
    get_labeled_beat_vector,
    _split_labeled_beat_vector,
    compute_rhythm_self_similarity_matrix,
    plot_rmss,
    compute_all_rmss,
    get_symbolic_length_classes,
    note_length_histogram,
    note_length_transition_matrix,
    plot_note_length_transition_matrix,
)
from .self_similarity import (
    compute_ssm,
    self_similarity_louie,
    self_similarity_single_measure,
    self_similarity_measures,
    plot_ssm,
    _self_similarity,
    binarize_self_similarity_matrix,
    feature_vector,
    get_novelty_func,
    get_segment_boundaries,
    plot_novelty_from_ssm,
)
from .graphs import (
    musa_to_graph,
    plot_graph,
)
from .structure import(
    PeltArgs,
    StructurePrediction,
    LevelsBPS,
    LevelsSWD,
)

__all__ = [
    "get_highest_lowest_pitches",
    "get_pitch_range",
    "get_pitch_classes",
    "get_note_density",
    "get_note_classes",
    "get_last_note_class",
    "get_chord_type_from_note_seq",
    "get_intervals_note_seq",
    "predict_chords",
    "predict_scales_degrees",
    "predict_possible_progressions",
    "predict_progression",
    "pitch_class_histogram",
    "pitch_class_transition_matrix",
    "get_symbolic_length_classes",
    "note_length_histogram",
    "note_length_transition_matrix",
    "plot_note_length_transition_matrix",
    "plot_pitch_class_transition_matrix",
    "pitch_counts",
    "average_pitch_interval",
    "_all_note_seq_permutations",
    "_delete_repeated_note_names",
    "_extract_note_positions",
    "_order_note_seq_by_chromatic_idx",
    "get_harmonic_density",
    "predic_time_sig_numerator",
    "get_start_sec",
    "get_ioi",
    "_delete_duplicates",
    "get_labeled_beat_vector",
    "_split_labeled_beat_vector",
    "compute_rhythm_self_similarity_matrix",
    "plot_rmss",
    "compute_all_rmss",
    "compute_ssm",
    "self_similarity_louie",
    "self_similarity_single_measure",
    "self_similarity_measures",
    "plot_ssm",
    "_self_similarity",
    "binarize_self_similarity_matrix",
    "feature_vector",
    "get_novelty_func",
    "get_segment_boundaries",
    "plot_novelty_from_ssm",
    "musa_to_graph",
    "plot_graph",
    "PeltArgs",
    "StructurePrediction",
    "LevelsBPS",
    "LevelsSWD",
]
