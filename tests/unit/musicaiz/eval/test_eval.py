import pytest
import matplotlib.pyplot as plt

from musicaiz import eval
from musicaiz.eval import _DEFAULT_MEASURES


@pytest.fixture
def dataset_path(fixture_dir):
    return fixture_dir / "datasets" / "jsbchorales" / "train"


@pytest.fixture
def dataset2_path(fixture_dir):
    return fixture_dir / "datasets" / "jsbchorales" / "test"


def test_get_all_dataset_measures(dataset_path):
    measures = eval.get_all_dataset_measures(
        dataset_path
    )
    assert len(measures) != 0


def test_get_average_dataset_measures(dataset_path):
    avgs = eval.get_average_dataset_measures(
        dataset_path
    )
    assert set(avgs.keys()) == set(_DEFAULT_MEASURES)


def test_get_distribution_all(dataset_path, dataset2_path):
    measures_1 = eval.get_all_dataset_measures(
        dataset_path
    )
    measures_2 = eval.get_all_dataset_measures(
        dataset2_path
    )

    # Compute the distances
    dataset_measures_dist = eval.euclidean_distance(measures_1)
    dataset2_measures_dist = eval.euclidean_distance(measures_2)
    inter_measures_dist = eval.euclidean_distance(measures_1, measures_2)

    keys = set(["0-1", "1-0", "1-2", "2-1", "0-2", "2-0"])
    assert set(dataset_measures_dist.keys()) == keys
    assert set(dataset2_measures_dist.keys()) == keys
    assert set(inter_measures_dist.keys()) == keys

    for v in dataset_measures_dist.values():
        set(v.keys()) == set(_DEFAULT_MEASURES)
    for v in dataset2_measures_dist.values():
        set(v.keys()) == set(_DEFAULT_MEASURES)
    for v in inter_measures_dist.values():
        set(v.keys()) == set(_DEFAULT_MEASURES)

    # Plot the distributions
    eval.get_distribution(
        (dataset_measures_dist, "Dataset 1 intra"),
        (dataset2_measures_dist, "Dataset 2 intra"),
        (inter_measures_dist, "Dataset 1 - Dataset 2 inter"),
        measure="all",
        show=False
    )
    plt.close('all')

    eval.model_features_violinplot(
        dataset_measures_dist, dataset2_measures_dist,
        "Dataset 1", "Dataset 2",
        show=False
    )
    plt.close('all')

    eval.plot_measures(
        dataset_measures_dist, dataset2_measures_dist,
        "Dataset 1", "Dataset 2",
        show=False
    )
    plt.close('all')

    # Compute the overlapping area between the distributions
    ov_area = eval.compute_overlapped_area(
        dataset_measures_dist, dataset2_measures_dist, "PR"
    )

    assert ov_area <= 1.0

    kld = eval.compute_kld(
        dataset_measures_dist, dataset2_measures_dist, "PR"
    )
