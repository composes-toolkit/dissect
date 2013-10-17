"""Test similarity measures."""
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.similarity.dot_prod import DotProdSimilarity
from composes.similarity.cos import CosSimilarity
from composes.similarity.euclidean import EuclideanSimilarity
from composes.similarity.lin import LinSimilarity
from composes.utils.matrix_utils import to_compatible_matrix_types

import pytest


@pytest.fixture
def vector_coordinates():
    """The coordinates of the vector."""
    return 1.0, 2.0, 3.0


@pytest.fixture
def other_vector_coordinates():
    """The coordinates of the other vector."""
    return 1.0, 2.0, 3.0


@pytest.fixture
def vector_type():
    """The type of the array."""
    return np.array


@pytest.fixture
def other_vector_type():
    """The type of the other array."""
    return np.array


@pytest.fixture
def vector(vector_type, vector_coordinates):
    """A vector."""
    # Some types expect some specific type instances, covert to an array first.
    return vector_type(np.array(vector_coordinates))


@pytest.fixture
def other_vector(other_vector_type, other_vector_coordinates):
    """Another vector."""
    # Some types expect some specific type instances, covert to an array first.
    return other_vector_type(np.array(other_vector_coordinates))


@pytest.fixture
def matrix_coordinates():
    """The coordinates of the matrix."""
    return (
        (1.0, 2.0, 3.0)
        (3.0, 2.0, 1.0),
    )


@pytest.fixture
def matrix(matrix_coordinates):
    """A matrix."""
    return np.array(matrix_coordinates)


@pytest.mark.parametrize(
    ('vector_type', 'other_vector_type', 'expected_matrix_type'),
    (
        (np.array, np.mat, DenseMatrix),
        (np.mat, csc_matrix, DenseMatrix),
        (np.mat, csr_matrix, DenseMatrix),
        (DenseMatrix, csc_matrix, DenseMatrix),
        (DenseMatrix, csr_matrix, DenseMatrix),
        (np.mat, SparseMatrix, SparseMatrix),
        (DenseMatrix, SparseMatrix, DenseMatrix),
        (SparseMatrix, DenseMatrix, SparseMatrix),
    )
)
def test_to_compatible_matrix_type(vector, other_vector, expected_matrix_type):
    """Test that two vectors are converted to a compatible type."""
    r1, r2 = to_compatible_matrix_types(vector, other_vector)

    assert isinstance(r1, expected_matrix_type)
    assert isinstance(r2, expected_matrix_type)


@pytest.mark.parametrize(
    ('similarity', 'vector_coordinates', 'other_vector_coordinates', 'expected_result'),
    (
        (DotProdSimilarity, (1.0, 2.0, 3.0), (1.0, 2.0, 3.0), 14.0),
        (DotProdSimilarity, (1.0, 2.0, 3.0), (0, -2.0, 3.0), 5.0),
        (DotProdSimilarity, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0), 0.0),

        (CosSimilarity, (1.0, 2.0, 3.0), (1.0, 2.0, 3.0), 1.0),
        (CosSimilarity, (1.0, 2.0, 3.0), (0, -2.0, 3.0), 0.3706246583305),
        (CosSimilarity, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0), 0.0),

        (LinSimilarity, (1.0, 2.0, 3.0), (1.0, 2.0, 3.0), 1.0),
        (LinSimilarity, (1.0, 2.0, 3.0), (0, -2.0, 3.0), 0.85714285714285),
        (LinSimilarity, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0), 0.0),

        (EuclideanSimilarity, (1.0, 2.0, 3.0), (1.0, 2.0, 3.0), 1.0),
        (EuclideanSimilarity, (1.0, 2.0, 3.0), (0, -2.0, 3.0), 0.19519410),
        (EuclideanSimilarity, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0), 0.21089672),
    )
)
def test_vector_similarity(similarity, vector, other_vector, expected_result):
    """Test the similarity measures between two vectors."""
    result = similarity().get_sim(vector, other_vector)

    assert np.allclose(result, expected_result)


@pytest.mark.parametrize(
    ('similarity', 'vector_coordinates', 'matrix_coordinates', 'expected_result'),
    (
        (
            DotProdSimilarity,
            (1.0, 2.0, 3.0),
            (
                (1.0, 2.0, 3.0),
            ),
            (
                (14.0, ),
            ),
        ),
        (
            DotProdSimilarity,
            (1.0, 2.0, 3.0),
            (
                (1.0, 2.0, 3.0),
                (1.0, 2.0, 3.0),
            ),
            (
                (14.0, ),
                (14.0, ),
            ),
        ),
        (
            DotProdSimilarity,
            (0, -2.0, 3.0),
            (
                (1.0, 2.0, 3.0),
                (1.0, 2.0, 3.0),
            ),
            (
                (5.0, ),
                (5.0, ),
            ),
        ),

        (
            DotProdSimilarity,
            (0.0, 0.0, 0.0),
            (
                (1.0, 2.0, 3.0),
                (1.0, 2.0, 3.0),
            ),
            (
                (0.0, ),
                (0.0, ),
            ),
        ),

        (
            CosSimilarity,
            (1.0, 2.0, 3.0),
            (
                (1.0, 2.0, 3.0),
            ),
            (
                (1.0, ),
            ),
        ),
        (
            CosSimilarity,
            (1.0, 2.0, 3.0),
            (
                (1.0, 2.0, 3.0),
                (1.0, 2.0, 3.0),
            ),
            (
                (1.0, ),
                (1.0, ),
            ),
        ),
        (
            CosSimilarity,
            (0, -2.0, 3.0),
            (
                (1.0, 2.0, 3.0),
                (1.0, 2.0, 3.0),
            ),
            (
                (0.370624, ),
                (0.370624, ),
            ),
        ),

        (
            CosSimilarity,
            (0.0, 0.0, 0.0),
            (
                (1.0, 2.0, 3.0),
                (1.0, 2.0, 3.0),
            ),
            (
                (0.0, ),
                (0.0, ),
            ),
        ),

        (
            LinSimilarity,
            (1.0, 2.0, 3.0),
            (
                (1.0, 2.0, 3.0),
            ),
            [[1.0]]),
        (
            LinSimilarity,
            (1.0, 2.0, 3.0),
            (
                (1.0, 2.0, 3.0),
                (1.0, 2.0, 3.0),
            ),
            (
                (1.0, ),
                (1.0, ),
            ),
        ),
        (
            LinSimilarity,
            (0, -2.0, 3.0),
            (
                (1.0, 2.0, 3.0),
                (1.0, 2.0, 3.0),
            ),
            (
                (0.857142, ),
                (0.857142, ),
            ),
        ),

        (
            LinSimilarity,
            (0.0, 0.0, 0.0),
            (
                (1.0, 2.0, 3.0),
                (1.0, 2.0, 3.0),
            ),
            (
                (0.0, ),
                (0.0, ),
            ),
        ),

        (
            EuclideanSimilarity,
            (1.0, 2.0, 3.0),
            (
                (1.0, 2.0, 3.0),
            ),
            (
                (1.0, ),
            ),
        ),
        (
            EuclideanSimilarity,
            (1.0, 2.0, 3.0),
            (
                (1.0, 2.0, 3.0),
                (1.0, 2.0, 3.0),
            ),
            (
                (1.0, ),
                (1.0, ),
            ),
        ),
        (
            EuclideanSimilarity,
            (0, -2.0, 3.0),
            (
                (1.0, 2.0, 3.0),
                (1.0, 2.0, 3.0),
            ),
            (
                (0.195194, ),
                (0.195194, ),
            ),
        ),

        (
            EuclideanSimilarity,
            (0.0, 0.0, 0.0),
            (
                (1.0, 2.0, 3.0),
                (1.0, 2.0, 3.0),
            ),
            (
                (0.210896, ),
                (0.210896, ),
            ),
        ),
    ),
)
def test_matrix_vector_similarity(similarity, vector, matrix, expected_result):
    """Test the similarity measures between a vector and a matrix."""
    result = similarity().get_sims_to_matrix(vector, matrix)

    np.testing.assert_array_almost_equal(np.mat(expected_result), result.mat)
