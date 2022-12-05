from pathlib import Path
from typing import List, Optional, Union, TextIO
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial, signal

from musicaiz.loaders import Musa
from musicaiz.structure import Note
from musicaiz import loaders
from musicaiz.utils import (
    group_notes_in_subdivisions_bars,
    get_highest_subdivision_bars_notes,
    __initialization
)
from musicaiz.eval import (
    get_eval_measures,
    _append_zeros_array,
    _DEFAULT_MEASURES
)


SSM_OPTIONS = ["louie", "measure", "all_measures"]



def compute_ssm(
    file: Union[Musa, str, TextIO, Path],
    ssm_type: str = "louie",
    measures: Optional[Union[List[str], str]] = None,
    is_binary: bool = True,
    threshold: float = 0.5
) -> np.ndarray:
    """Computes the selected SSM.
    
    Parameters
    ----------

    file: Union[Musa, str, TextIO, Path]
        A file or a `musicaiz` :func:`~musicaiz.loaders.Musa` object.

    ssm_type: str
        Valid values: ``louie``, ``measure`` , ``measures``.

        - ``louie``: Computes the SSM with Louie's method :func:`~musicaiz.features.self_similarity_louie`
        
        - ``measure``: Computes the SSM with one of the :const:`~musicaiz.eval._DEFAULT_MEASURES` measures
        :func:`~musicaiz.features.self_similarity_single_measure`.
        Note that if this ``ssm_type`` is selected, the ``measures`` argument must be a valid measure str.
        
        - ``measures``: Computes the SSM with more than one of the :const:`~musicaiz.eval._DEFAULT_MEASURES`
        measures :func:`~musicaiz.features.self_similarity_measures`.
        Note that if this ``ssm_type`` is selected, the ``measures`` argument must be a list of valid
        measure strs or ``all``.

    measures: Union[List[str], str]
        The measure or list of measures to compute the SSM with. It also accepts ``all`` as a value if
        ``ssm_type`` is set to ``measures``.

    is_binary: bool = True
        Binarizes the SSM.

    threshold: int = 0.5
        The threshold value used to binarize the SSSM.
    
    Returns
    -------
    s: np.ndarray [shape=(n_bars, n_bars)]
        The SSM computed from the feature_vector or measure(s).
    
    Examples
    --------

    >>> from pathlib import Path
    >>> file = Path("tests/fixtures/tokenizers/mmm_tokens.mid")
    >>> ssm = musicaiz.features.compute_ssm(
        file=file,
        ssm_type="measures",
        measures="all"
    )
    """
    musa_obj = __initialization(file)      
    if ssm_type == "louie":
        s = self_similarity_louie(musa_obj)
    elif ssm_type == "measure":
        if not isinstance(measures, str):
            raise ValueError("Measures arg for `measure` SSM must be a valid str measure.")
        s = self_similarity_single_measure(musa_obj, measures)
    elif ssm_type == "measures":
        s = self_similarity_measures(musa_obj, measures)
    else:
        raise ValueError(f"Select a valid SSM: {SSM_OPTIONS}")
    
    if is_binary:
        if threshold < 0 or threshold > 1:
            raise ValueError(f"Threshold value {threshold} is not between 0 and 1.")
        s = binarize_self_similarity_matrix(s, threshold)
    return s


def self_similarity_louie(file: Union[Musa, str, TextIO, Path]) -> np.ndarray:
    """
    Computes the SSM with Louie method:
    https://wlouie1.github.io/MusicPlot/musicplot_paper.pdf

    Parameters
    ----------

    file: Union[Musa, str, TextIO, Path]
        A file or a `musicaiz` :func:`~musicaiz.loaders.Musa` object.

    Returns
    -------

    s: np.ndarray [shape=(n_bars, n_bars)]
        The SSM computed from the ``feature_vector``.
    """
    musa_obj = __initialization(file)
    all_subdiv_notes = musa_obj.get_notes_in_beats(0, len(musa_obj.beats) - 1)
    bar_highest_subdiv_notes = get_highest_subdivision_bars_notes(all_subdiv_notes)
    m = feature_vector(bar_highest_subdiv_notes)
    s = _self_similarity(m)
    return s


def self_similarity_single_measure(
    file: Union[Musa, str, TextIO, Path],
    measure: str
) -> np.ndarray:
    """
    Computes the SSM with the selected measure for all the bars.

    Parameters
    ----------

    file: Union[Musa, str, TextIO, Path]
        A file or a `musicaiz` :func:`~musicaiz.loaders.Musa` object.

    measure: str
        The measure or feature used to compute the SSM.

    Returns
    -------

    s: np.ndarray [shape=(n_bars, n_bars)]
        The SSM computed from the selected measure.
    """
    musa_obj = __initialization(file)
    bar_measures = []
    for b in musa_obj.bars:
        bar = musa_obj.get_notes_in_bar(b)
        bar_measures.append(get_eval_measures(bar, measures=[measure]))
    s = np.zeros((len(bar_measures), len(bar_measures)))
    for i in range(len(bar_measures)):
        for j in range(len(bar_measures)):
            # Flatten vector
            if isinstance(bar_measures[i][measure], np.ndarray):
                m_i = bar_measures[i][measure].flatten()
            else:
                m_i = bar_measures[i][measure]
            if isinstance(bar_measures[j][measure], np.ndarray):
                m_j = bar_measures[j][measure].flatten()
            else:
                m_j = bar_measures[j][measure]

            # Append zeros to any vector if shapes do not match
            if isinstance(m_i, np.ndarray) or isinstance(m_j, np.ndarray):
                m_i, m_j = _append_zeros_array(m_i, m_j)

            s[i, j] = spatial.distance.cosine(m_i, m_j)
    return s


def self_similarity_measures(
    file: Union[Musa, str, TextIO, Path],
    measures: List[str]
) -> np.ndarray:
    """Computes the SSM with the selected measure for all the bars.

    Parameters
    ----------

    file: Union[Musa, str, TextIO, Path]
        A file or a `musicaiz` :func:`~musicaiz.loaders.Musa` object.

    measure: str
        The list of measures used to compute the SSM. It also accepts ``all`` as a value if
        we want to compute the SSM with all the available measures.

    Returns
    -------

    s: np.ndarray [shape=(n_bars, n_bars)]
        The SSM computed from the selected measures.
    """
    musa_obj = __initialization(file)

    if isinstance(measures, str) and not measures == "all":
        raise ValueError("For input arg `measures` you must provide a list of valid measures or `all`.")
    elif isinstance(measures, str) and measures == "all":
        measures = _DEFAULT_MEASURES
    elif not isinstance(measures, list) and not all(m in _DEFAULT_MEASURES for m in measures):
        raise ValueError(f"Measures arg must be a list of valid measures {_DEFAULT_MEASURES}.")

    musa_obj = __initialization(file)
    bar_measures = []
    for b in musa_obj.bars:
        bar = musa_obj.get_notes_in_bar(b)
        bar_measures.append(get_eval_measures(bar, measures=measures))
    s = np.zeros((len(bar_measures), len(bar_measures)))
    for i in range(len(bar_measures)):
        for j in range(len(bar_measures)):
            # Flatten vector
            m_sim = 0
            for measure in bar_measures[i].keys():
                if isinstance(bar_measures[i][measure], np.ndarray):
                    m_i = bar_measures[i][measure].flatten()
                else:
                    m_i = bar_measures[i][measure]
                if isinstance(bar_measures[j][measure], np.ndarray):
                    m_j = bar_measures[j][measure].flatten()
                else:
                    m_j = bar_measures[j][measure]

                # Append zeros to any vector if shapes do not match
                if isinstance(m_i, np.ndarray) or isinstance(m_j, np.ndarray):
                    m_i, m_j = _append_zeros_array(m_i, m_j)

                m_sim = spatial.distance.cosine(m_i, m_j)
                m_sim += m_sim
            # mean of the cosine distances
            s[i, j] = m_sim / len(bar_measures[i].keys())
    return s


def plot_ssm(
    ssm: np.ndarray,
    segments: bool = False,
    threshold: float = 0.5,
    window: float = 2,
    save: Optional[bool] = False,
    filename: str = "ssm",
    origin: str = "lower",
    title: Optional[str] = None,
    colorbar: bool = True,
    dpi: int = 300
):
    """
    Plots a SSM.

    Parameters
    ----------

    ssm: np.ndarray [shape=(n_bars, n_bars)]
        The SSM.

    segments: bool
        Plot the sections boundaries over the SSM by computing the
        novelty function. Default is False.

    threshold: float
        The threshold for peak picking in the novelty curve if ``segments=True``.
        Default is 0.5.

    window: float
        The window for peak picking in the novelty curve if ``segments=True``.
        Default is 2.

    save: Optional[bool]
        True if we want to save the output plot in disk. Defaults to True.

    filename: str
        The name of the file of the plot in disk. Defaults to ``"ssm"``.

    origin: str
        to matplotlib.pyplot.imshow: https://matplotlib.org/3.5.0/tutorials/intermediate/imshow_extent.html.
        Defaults to ``"lower"``.

    title: Optional[str]
        Plot title. Defaults to ``None``.

    colorbar: bool
       ``True`` if we want to show the colorbar in the plot. Defaults to ``True``.

    dpi: int
        dpi of the plot. Defaults to ``300``.

    Examples
    --------

    >>> from pathlib import Path
    >>> file = Path("C:/Users/Carlos/Downloads/988-aria.mid")
    >>> ssm = musicaiz.features.compute_ssm(
        file=file,
        ssm_type="measures",
        measures="all"
    )
    >>> plot_ssm(ssm)
    """
    if title is not None:
        plt.title(title)
    plt.imshow(ssm, origin=origin)
    if save:
        plt.savefig(filename + ".png")
    if colorbar:
        plt.colorbar()
    if segments:
        segment_boundaries = get_segment_boundaries(ssm, threshold, window)
        for i in range(len(segment_boundaries)):
            plt.axvline(segment_boundaries[i], color='w', linestyle='--', alpha=.5)
            plt.axhline(segment_boundaries[i], color='w', linestyle='--', alpha=.5)
    plt.gcf().set_dpi(dpi)
    plt.show()


def _self_similarity(feature_vector: List[List[Note]]) -> np.ndarray:
    """Converts a feature vector with the highest notes per subdivision and bar
    in a Self-Similarity Matrix with the  cosine distance.

    Parameters
    ----------

    feature_vector: List[List[Note]]
        A vector with all bars and subdivisions in which each element represents
        the note with the highest pitch.

    Returns
    -------

    s: np.ndarray 2d
        The SSM computed from the feature_vector.
    """
    s = np.zeros((len(feature_vector), len(feature_vector)))
    for i in range(len(feature_vector)):
        for j in range(len(feature_vector)):
            # Flatten lists to convert them in 1D vectors
            m_i = list(sum(feature_vector[i] ,()))
            m_j = list(sum(feature_vector[j] ,()))
            s[i, j] = spatial.distance.cosine(m_i, m_j)
    return s


def binarize_self_similarity_matrix(
    ssm: np.ndarray,
    threshold: float
) -> np.ndarray:
    bin_s = np.zeros((ssm.shape[0], ssm.shape[0]))
    for i in range(ssm.shape[0]):
        for j in range(ssm.shape[0]):
            if i == j:
                bin_s[i, j] = 1
                continue
            if ssm[i, j] >= threshold:
                bin_s[i, j] = 1
            else:
                bin_s[i, j] == 0
    return bin_s


def feature_vector(bar_highest_subdiv_notes: List[List[Note]]) -> List[List[Note]]:
    """
    Computes the m_prim vector which calculates the difference between 2 consecutive note's
    pitches and the division of the note's durations.

    Parameters
    ----------

    bar_highest_subdiv_notes: List[List[Note]]
        A list of bars in which each element in the bar corresponds to the note with the
        highest pitch in the subdivision.

    Returns
    -------

    m_prime: List[List[Note]]
        A list ob bars in in which each element in the bar corresponds to the tuple
        (p2-p1, d2/d1) with p the pitch and d the duration of the note.
    """
    m_prime = []
    for bar in bar_highest_subdiv_notes:
        bar_m = []
        for i, note in enumerate(bar):
            if i + 1 >= len(bar):
                break
            y = bar[i].pitch - bar[i + 1].pitch
            if bar[i].start_ticks != 0:
                x = (bar[i + 1].end_ticks - bar[i + 1].start_ticks) / (bar[i].end_ticks - bar[i].start_ticks)
            else:
                x = 0
            bar_m.append((y, x))
        m_prime.append(bar_m)
    return m_prime


def get_novelty_func(
    ssm: np.ndarray,
    is_normalized: bool = True
) -> np.ndarray:
    """
    Computes the novelty function of a SSM.

    Parameters
    ----------

    ssm: np.ndarray [shape=(n_bars, n_bars)]
        The SSM.

    is_normalized: bool
        Normalizes the Novelty curve. Defaults is True.

    Returns
    -------

    c: np.ndarray [shape=(n_bars,)]
        The normalized novelty function as a 1D array.
    """

    c = np.linalg.norm(ssm[:, 1:] - ssm[:, 0:-1], axis=0)

    if is_normalized:
        c_norm = (c - c.min()) / (c.max() - c.min())  # normalization of c
        return c_norm
    else:
        return c


def get_segment_boundaries(
    ssm: np.ndarray,
    threshold: float = 0.5,
    window: float = 2,
) -> np.ndarray:
    """
    Gets the segment boundaries of a SSM.

    Parameters
    ----------

    ssm: np.ndarray [shape=(n_bars, n_bars)]
        The SSM.

    threshold: float
        The threshold for peak picking in the novelty curve if ``segments=True``.
        Default is 0.5.

    window: float
        The window for peak picking in the novelty curve if ``segments=True``.
        Default is 2.

    Returns
    -------

    novelty: np.ndarray [shape=(n_bars,)]
        The novelty function as a 1D array.
    """
    # TODO: Output not only the bar indexes but add option to also output secs or ticks
    # where boundaries are (this is simple to compute with the rhythm submodule.)
    novelty = get_novelty_func(ssm=ssm, is_normalized=True)

    # Peaks detection - sliding window
    lamda = round(window)  # window length
    peaks_position = signal.find_peaks(
        novelty,
        height=threshold,
        distance=lamda,
        width=round(threshold)
    )[0]  # array of peaks
    peaks_values = signal.find_peaks(
        novelty,
        height=threshold,
        distance=lamda,
        width=round(threshold)
    )[1]['peak_heights']  #array of peaks
    b = peaks_position

    # Adding elements 1 and N' to the begining and end of the array
    if b[0] != 0:
        b = np.concatenate([[0], b])  # b: segment boundaries
    if b[-1] != ssm.shape[0] - 1:
        b = np.concatenate([b, [ssm.shape[0] - 1]])

    return b


def plot_novelty_from_ssm(
    ssm: np.ndarray,
    segments: bool = False,
    threshold: float = 0.5,
    window: float = 2,
    save: Optional[bool] = False,
    filename: str = "ssm",
    title: Optional[str] = None,
    dpi: int = 300
):
    """
    Plots the novelty curve from a SSM.

    Parameters
    ----------

    ssm: np.ndarray [shape=(n_bars, n_bars)]
        The SSM.

    segments: bool
        Plot the sections boundaries over the SSM by computing the
        novelty function. Default is False.

    threshold: float
        The threshold for peak picking in the novelty curve if ``segments=True``.
        Default is 0.5.

    window: float
        The window for peak picking in the novelty curve if ``segments=True``.
        Default is 2.

    save: Optional[bool]
        True if we want to save the output plot in disk. Defaults to True.

    filename: str
        The name of the file of the plot in disk. Defaults to ``"ssm"``.

    title: Optional[str]
        Plot title. Defaults to ``None``.

    dpi: int
        dpi of the plot. Defaults to ``300``.
    """
    # Plot novelty function with boundaries
    plt.figure(figsize=(50, 10), dpi=dpi)
    if title is not None:
        plt.title(title)
    if save:
        plt.savefig(filename + ".png")
    plt.title(title)

    novelty = get_novelty_func(ssm)
    frames = range(len(novelty))
    if segments:
        segment_boundaries = get_segment_boundaries(ssm, threshold, window)
        for i in range(len(segment_boundaries)):
            plt.axvline(segment_boundaries[i], color='r', linestyle='--')
    plt.plot(frames, novelty)
    plt.show()
