import csv 
import itertools
import numpy as np

import pytest

from jigsaw.utils.permutations import hamming_distance_matrix, index_from_distance_matrix, create_permutations


def test_hamming_distance_matrix():
    x = np.array([[1, 2, 3], [2, 3, 4], [4, 3, 2], [1, 2, 2]])
    y = np.array([1, 2, 3])

    expected_distance = np.array([0, 3, 3, 1]) 
    distance = hamming_distance_matrix(x, y)

    assert expected_distance == distance

def test_index_from_distance_matrix():
    distance_matrix = np.array([[0, 1, 2, 3, 2, 8],
                                 [1, 8, 8, 8, 8, 8],
                                 [3, 8, 2, 3, 8, 8],
                                 [4, 8, 8, 3, 2, 8]])

    expected_index = 5
    index = index_from_distance_matrix(distance_matrix)

    assert expected_index == index

def test_create_permutations():
    permutations = create_permutations(5)
    for i in range(permutations.shape[0]):
        assert hamming_distance_matrix(permutations[i, :], permutations[:, :]).sum() == 4 * 8
