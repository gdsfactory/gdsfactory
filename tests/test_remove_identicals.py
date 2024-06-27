import pytest

from gdsfactory.functions import remove_identicals


def test_single_point(self):
    pts = np.array([[0, 0]])
    result = remove_identicals(pts)
    np.testing.assert_array_equal(result, pts)


def test_no_identicals_closed(self):
    pts = np.array([[0, 0], [1, 1], [2, 2]])
    result = remove_identicals(pts, closed=True)
    np.testing.assert_array_equal(result, pts)


def test_no_identicals_not_closed(self):
    pts = np.array([[0, 0], [1, 1], [2, 2]])
    result = remove_identicals(pts, closed=False)
    np.testing.assert_array_equal(result, pts)


def test_identicals_closed(self):
    pts = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
    expected_result = np.array([[0, 0], [1, 1]])
    result = remove_identicals(pts, closed=True)
    np.testing.assert_array_equal(result, expected_result)


def test_identicals_not_closed(self):
    pts = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
    expected_result = np.array([[0, 0], [1, 1], [1, 1]])
    result = remove_identicals(pts, closed=False)
    np.testing.assert_array_equal(result, expected_result)


def test_identicals_mixed(self):
    pts = np.array([[0, 0], [0, 0], [1, 1], [2, 2], [2, 2], [3, 3]])
    expected_result_closed = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    expected_result_not_closed = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

    result_closed = remove_identicals(pts, closed=True)
    result_not_closed = remove_identicals(pts, closed=False)

    np.testing.assert_array_equal(result_closed, expected_result_closed)
    np.testing.assert_array_equal(result_not_closed, expected_result_not_closed)


if __name__ == "__main__":
    pytest.main()
