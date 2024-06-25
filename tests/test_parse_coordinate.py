import pytest

from gdsfactory.component_layout import _parse_coordinate


def test_object_with_center():
    class MockObject:
        def __init__(self, center):
            self.center = center
            self.dcenter = (1.5, 2.5)

    obj = MockObject(center=True)
    result = _parse_coordinate(obj)
    assert result == obj.dcenter


def test_array_like_with_two_elements():
    coordinate = [1.0, 2.0]
    result = _parse_coordinate(coordinate)
    assert result == coordinate


def test_array_like_with_incorrect_number_of_elements():
    coordinate = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        _parse_coordinate(coordinate)
