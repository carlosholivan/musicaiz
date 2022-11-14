"""
Evaluation
==========

This module evaluates music generation systems.


Music Generation
----------------

This module contains the implementation of the paper:

.. panels::

    [1] Yang, L. C., & Lerch, A. (2020).
    On the evaluation of generative models in music.
    Neural Computing and Applications, 32(9), 4773-4784.
    https://doi.org/10.1007/s00521-018-3849-7


.. autosummary::
    :toctree: generated/

    PitchMeasures
    RhythmMeasures
    get_all_dataset_measures
    get_average_dataset_measures
    get_eval_measures
    euclidean_distance
    get_distribution
    compute_overlapped_area
    compute_kld
    compute_oa_kld
    plot_measures
    model_features_violinplot

"""

import copy
from enum import Enum
from typing import Dict, Union, List, Optional, Any
import numpy as np
import scipy
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import seaborn as sns
import pandas as pd
from pathlib import Path


from musicaiz import features, loaders
from musicaiz.structure import Note
from musicaiz import wrappers


class PitchMeasures(Enum):
    PC = "Pitch Counts"
    PCH = "Pitch Class Histogram"
    PR = "Pitch Range"
    PCTM = "Pitch Class Transition Matrix"
    PI = "Average Pitch Interval"


class RhythmMeasures(Enum):
    NC = "Note Counts"
    IOI = "Inter-Onset Intervals"
    NLH = "Note Length Histogram"
    NLTM = "Note Length Transition Matrix"


_DEFAULT_MEASURES = PitchMeasures._member_names_ + RhythmMeasures._member_names_
_MARKERS = ["o", "^", "s", "D"]
_COLORS = [
    "#40E0D0",
    "#CCCCFF",
    "#DE3163",
    "#FFBF00",
    "#6495ED",
    "#FF7F50",
    "#9FE2BF",
    "#000080",
    "#FF00FF",
    "#800000",
    "#FFFF00",
    "#C0C0C0"
]


@wrappers.timeis
def get_all_dataset_measures(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Returns the measures of each midi file in the directory or
    subdirectoy `dataset_path`."""

    midi_measures = wrappers.multiprocess_path(
        func=_eval_file,
        path=path,
        args=None
    )

    return midi_measures


@wrappers.timeis
def get_average_dataset_measures(path: Union[str, Path]):
    """Returns the measures of each midi file in the directory or
    subdirectoy `dataset_path`.

    Returns
    -------
    avgs: Dict[str, Union[int, np.array]]
        the average measures of the dataset where each key in the
        dictionary is an evaluation feature.
    """

    midi_measures = get_all_dataset_measures(path)

    measures = {}
    for key in midi_measures[0].keys():
        measures.update({key: [d[key] for d in midi_measures]})

    # IOIs may not have the same length
    prev_ioi_len = 0
    for ioi in measures["IOI"]:
        if len(ioi) > prev_ioi_len:
            max_len = len(ioi)
            prev_ioi_len = max_len
    iois = copy.deepcopy(measures["IOI"])
    for i, ioi in enumerate(iois):
        if len(ioi) < max_len:
            diff = max_len - len(ioi)
            measures["IOI"] = np.pad(ioi, (0, diff))

    avgs = {}
    for key in measures.keys():
        avgs.update({key: sum(measures[key]) / len(measures[key])})
    return avgs


def _eval_file(file: Union[str, Path]):
    midi = loaders.Musa(file)
    midi_measure = get_eval_measures(midi.notes)
    return midi_measure


def get_eval_measures(
    notes: List[Note],
    measures: List[str] = _DEFAULT_MEASURES
) -> Dict[str, Union[int, np.array]]:

    """Computes the features used to evaluate music.

    Parameters
    ----------
    notes: List[Note]

    measures: List[str]
        The measures that we want to compute.
        Ex.: ["PC, "PR"]

    Returns
    -------
    measures_dict: Dict[str, Union[int, np.array]]
        Each measure stores in a dict which keys are the measures' capitals.

    """
    measures_dict = {}

    # pitch counts (PC)
    if "PC" in measures:
        measures_dict["PC"] = features.pitch_counts(notes)

    # pitch class histogram
    if "PCH" in measures:
        measures_dict["PCH"] = features.pitch_class_histogram(notes)

    # pitch class transition matrix
    if "PCTM" in measures:
        measures_dict["PCTM"] = features.pitch_class_transition_matrix(notes)

    # pitch range (PR)
    if "PR" in measures:
        measures_dict["PR"] = features.get_pitch_range(notes)

    # average pitch interval (PI)
    if "PI" in measures:
        measures_dict["PI"] = features.average_pitch_interval(notes)

    # note counts (NC)
    if "NC" in measures:
        measures_dict["NC"] = len(notes)

    # inter-onset intervals (IOI)
    if "IOI" in measures:
        all_note_on = features.get_start_sec(notes)
        measures_dict["IOI"] = np.array(features.get_ioi(all_note_on))

    # note length histogram (NLH)
    if "NLH" in measures:
        measures_dict["NLH"] = features.note_length_histogram(notes)

    # note length transition matrix (NLTM)
    if "NLTM" in measures:
        measures_dict["NLTM"] = features.note_length_transition_matrix(notes)

    return measures_dict


def euclidean_distance(
    measures1: List[Dict[str, Union[int, np.array]]],
    measures2: Optional[List[Dict[str, Union[int, np.array]]]] = None,
) -> Dict[str, Dict[str, Union[int, np.array]]]:
    """Computes the pair-wise cross-validation with Euclidean distance of the
    features extracted from a dataset which are stored in a dict. Each item in
    the list is a dict of features corresponding to the features extracted from
    a list of notes.

    Parameters
    ----------
    measures: List[Dict[str, Union[int, np.array]]]
        A list of the features (stored in dicts) that have been
        extracted with the `get_eval_measures` method from different notes' lists.

    Returns
    -------
    all_dist: Dict[str, Dict[str, Union[int, np.array]]]
        The euclidean distance of the cross-validated measures.

    Example
    -------
    >>> measures = [{"PC": 5, "PR": 4}, {"PC": 7, "PR": 5}]
    >>> euclidean_distance(measures)
    >>> {"0-1": {"PC": 2.0, "PR": 1.0}}
    """

    # if no measures2 is passed, we'd measure the intra-set distances
    if measures2 is None:
        measures2 = measures1
    if len(measures1) <= 1:
        raise Exception("There is only one file in the dataset 1. There must be more than 1 file to measure the distances.")
    if len(measures2) <= 1:
        raise Exception("There is only one file in the dataset 1. There must be more than 1 file to measure the distances.")
    # initialize dataframe
    count = 0
    all_dist = {}
    for i in range(len(measures1)):
        for j in range(len(measures2)):
            if i != j:
                count += 1
                key = str(i) + "-" + str(j)
                dist = {key: {}}
                for k, v in measures2[j].items():
                    # if the feature is a vector, to compute the distance
                    # both vectors must be the same length so we append
                    # zeros to the vector with the lowest length
                    vec1 = measures1[i][k]
                    vec2 = measures2[j][k]
                    if type(vec1).__module__ == np.__name__:
                        vec1, vec2 = _append_zeros_array(vec1, vec2)
                    measure = {
                        k: np.linalg.norm(vec1 - vec2)
                    }
                    dist[key].update(measure)
                all_dist.update(dist)
    return all_dist


def _get_measures_to_plot(
    measures_dict,
    measure: str = "all"
):
    """Select which measures (or features) to plot."""
    if measure == "all":
        for v in measures_dict.values():
            measures = v.keys()
    else:
        measures = [measure]
    return measures


def get_distribution(
    *args,
    measure: str = "all",
    show: bool = True
):
    """
    Plots the measure histogram of the input distances
    datasets (args).

    Parameters
    ----------

    args: Tuple
        tuple(euclidean_distance dict, label str for the legend)

    measure: str
        the measure to plot its histogram of distances.
        If measure is `"all"` we'll get the plots of all the
        measures in the input dicts.
    """
    # plot all measures
    measures = _get_measures_to_plot(args[0][0], measure)

    # one plot per measure
    for i, measure in enumerate(measures):
        fig, ax = plt.subplots()
        labels = []
        for arg in args:
            # arg[0] is the dict, arg[1] is the label
            labels.append(arg[1])
            dist = _get_array_from_dict(arg[0], measure)
            sns.kdeplot(dist, shade=True, alpha=.1, ax=ax)
        if measure in PitchMeasures._member_names_:
            ax.set_title(PitchMeasures[measure].value)
        elif measure in RhythmMeasures._member_names_:
            ax.set_title(RhythmMeasures[measure].value)
        plt.legend(labels)
        if show:
            plt.show()


def _append_zeros_array(vec1, vec2):
    diff = abs(len(vec1) - len(vec2))
    if len(vec1) < len(vec2):
        vec1 = np.pad(vec1, (0, diff))
    if len(vec1) > len(vec2):
        vec2 = np.pad(vec2, (0, diff))
    assert len(vec1) == len(vec2)
    return vec1, vec2


def compute_overlapped_area(eucl_dist1, eucl_dist2, measure):
    dist1 = _get_array_from_dict(eucl_dist1, measure)
    dist2 = _get_array_from_dict(eucl_dist2, measure)
    x, kde0_x, kde1_x, inters_x = _get_intersections(dist1, dist2)
    # Integrate over intersected area
    ov_area = np.trapz(inters_x, x)
    return ov_area


def compute_kld(eucl_dist1, eucl_dist2, measure):
    dist1 = _get_array_from_dict(eucl_dist1, measure)
    dist1 = dist1 / np.linalg.norm(dist1)
    dist2 = _get_array_from_dict(eucl_dist2, measure)
    dist2 = dist2 / np.linalg.norm(dist2)
    dist1, dist2 = _append_zeros_array(dist1, dist2)
    kld = scipy.stats.entropy(pk=dist1, qk=dist2, base=None)
    return kld


def compute_oa_kld(
    measures_dist1: Dict[str, Dict[str, float]],
    measures_dist2: Dict[str, Dict[str, float]],
    label1: str = "1",
    label2: str = "2",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Computes the KLD and OA of all the features in the input
    dicts and build a dict with the reults.

    Parameters
    ----------

    measures_dist1: Dict[str, Dict[str, float]]
        the dict with the cross-validated distances of each sample
        in the database (intra or inter) for each feature.

    measures_dist2: Dict[str, Dict[str, float]]
        the dict with the cross-validated distances of each sample
        in the database (intra or inter) for each feature.


    Returns
    -------

    measures_dict: Dict[str, Dict[str, Dict[str, float]]]
        the output results stored in a dictionary in which the keys
        are `label1` and `label2` and the values are dicts which keys
        are the measures computed and the values are the KLD and OA.
    """
    measures_dict = {}
    names = [label1, label2]
    measures = _get_measures_to_plot(measures_dist1, "all")

    for i in range(0, 2, 1):
        measure_vals = {names[i]: {}}
        for measure in measures:
            # calculate oa and kld
            kld = compute_kld(
                measures_dist1, measures_dist2, measure
            )
            ov_area = compute_overlapped_area(
                measures_dist1, measures_dist2, measure
            )
            oa_kld = {measure: {"KLD": kld, "OA": ov_area}}
            measure_vals[names[i]].update(oa_kld)
        measures_dict.update(measure_vals)
    return measures_dict


def plot_measures(
    measures_dist1: Dict[str, Dict[str, float]],
    measures_dist2: Dict[str, Dict[str, float]],
    label1: str = "1",
    label2: str = "2",
    show: bool = True
):
    measures_dict = compute_oa_kld(
        measures_dist1,
        measures_dist2,
        label1,
        label2
    )
    fig, ax = plt.subplots()

    datasets = []
    points = []
    k1s = []
    i, j = 0, 0
    custom_lines = []
    # loop in datasets
    for k, v in measures_dict.items():
        k_prev = ""
        marker = _MARKERS[i]
        dataset = k + " "
        datasets.append(dataset)
        # loop in measures
        x, y = [], []
        for k1, v1 in v.items():
            # equal measures have the same color
            if k1 in k1s:
                idx = [k for k, kval in enumerate(k1s) if k1==kval]
                color = _COLORS[idx[0]]
            else:
                color = _COLORS[j]
                j += 1
            k1s.append(k1)
            # save point in list to later add legend with marker of database
            if k_prev != k:
                points += ax.plot(
                    v1["KLD"], v1["OA"],
                    color=color, marker=marker,
                    markersize=8, label=dataset, linestyle="None"
                )
            else:
                ax.plot(
                    v1["KLD"], v1["OA"],
                    color=color, marker=marker,
                    markersize=8, label=dataset, linestyle="None"
                )
            k_prev = k
        i += 1
    for index, key in enumerate(measures_dict):
        if index == 0:
            for idx_mes, key_mes in enumerate(measures_dict[key]):
                x, y = _get_points(measures_dict, key_mes)
                line = Line2D(x, y, color=_COLORS[idx_mes], lw=2, label=key_mes, linestyle="--")
                ax.add_line(line)
                custom_lines.append(line)
    leg = Legend(
        ax, points, [k for k in measures_dict.keys()],
        frameon=False, loc='lower right', markerscale=.5
    )
    ax.add_artist(leg)
    ax.legend(custom_lines, k1s)
    plt.ylabel("OA")
    plt.xlabel("KLD")
    if show:
        plt.show()


def model_features_violinplot(
    measures_dist1: Dict[str, Dict[str, float]],
    measures_dist2: Dict[str, Dict[str, float]],
    label1: str = "1",
    label2: str = "2",
    measure: str = "all",
    show: bool = True
):
    """Plots the distances distributions of the input
    features as a violin plot as the Fig. 4 of the paper
    by Yang and Lerch.
    Note that we normalize the distances vectors to plot them.

    Parameters
    ----------
    measures_dist1: Dict[str, Dict[str, float]]
        the dict with the cross-validated distances of each sample
        in the database (intra or inter) for each feature.

    measures_dist2: Dict[str, Dict[str, float]]
        the dict with the cross-validated distances of each sample
        in the database (intra or inter) for each feature.

    label1: str
        the label or name of the dataset of model which corresponds to
        the `measures_dict1`, for showing it in the legend.

    label2: str
        the label or name of the dataset of model which corresponds to
        the `measures_dict2`, for showing it in the legend.
    """
    fig, ax = plt.subplots()

    measures = _get_measures_to_plot(measures_dist1, measure)

    df_all = []
    for measure in measures:
        dist1 = _get_array_from_dict(measures_dist1, measure)
        dist1 = dist1 / np.linalg.norm(dist1)
        df1 = pd.DataFrame(
            {
                "distance": dist1,
                "feature": [measure for _ in range(len(dist1))],
                "model": [label1 for _ in range(len(dist1))],
            }
        )

        dist2 = _get_array_from_dict(measures_dist2, measure)
        dist2 = dist2 / np.linalg.norm(dist2)
        df2 = pd.DataFrame(
            {
                "distance": dist2,
                "feature": [measure for _ in range(len(dist2))],
                "model": [label2 for _ in range(len(dist2))],
            }
        )

        df = pd.concat([df1, df2])
        df_all.append(pd.concat([df]))
    df_all = pd.concat(df_all)
    sns.violinplot(
        data=df_all, x="feature", y="distance", hue="model",
        palette="coolwarm", split=True, ax=ax
    )
    plt.legend(frameon=False)
    ax.set_ylabel("Distance")
    ax.set_xlabel("Feature")
    if show:
        plt.show()


def _get_array_from_dict(eucl_dist_dict, measure):
    arr = []
    for k, v in eucl_dist_dict.items():
        arr.append(v[measure])
    return np.asarray(arr)


def _get_kde(dist1, dist2):
    kde0 = gaussian_kde(dist1, bw_method=0.3)
    kde1 = gaussian_kde(dist2, bw_method=0.3)
    return kde0, kde1


def _get_min_max(dist1, dist2):
    xmin = min(dist1.min(), dist2.min())
    xmax = max(dist1.max(), dist2.max())
    return xmin, xmax


def _get_min_max_with_margin(dist1, dist2, margin=0.2):
    xmin, xmax = _get_min_max(dist1, dist2)
    # add a 20% margin, as the kde is wider than the data
    dx = margin * (xmax - xmin)
    xmin -= dx
    xmax += dx
    return xmin, xmax


def _get_intersections(dist1, dist2):
    kde0, kde1 = _get_kde(dist1, dist2)
    # Get max and min between both distributions
    xmin, xmax = _get_min_max_with_margin(dist1, dist2)
    # build line from xmin to xmax
    x = np.linspace(xmin, xmax, 500)
    # Find intersections
    kde0_x = kde0(x)
    kde1_x = kde1(x)
    inters_x = np.minimum(kde0_x, kde1_x)
    return x, kde0_x, kde1_x, inters_x


def _get_points(measures_dict, measure="PC"):
    kld, ov_area = [], []
    for k, v in measures_dict.items():
        for k1, v1 in v.items():
            if k1 == measure:
                kld.append(v1["KLD"])
                ov_area.append(v1["OA"])
    return kld, ov_area
