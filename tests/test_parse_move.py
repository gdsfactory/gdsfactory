import pytest
import numpy as np
from gdsfactory.component_layout import _parse_move

def test_destination_none():
    origin = [3, 4]
    destination = None
    axis = 'x'
    dx, dy = _parse_move(origin, destination, axis)
    assert dx == 3, f"Expect dx to be 3, but got {dx}"
    assert dy == 4, f"Expect dy to be 4, but got {dy}"

def test_axis_x():
    origin = [1, 2]
    destination = [3, 4]
    axis = 'x'
    dx, dy = _parse_move(origin, destination, axis)
    assert dx == 2, f"Expect dx to be 2, but got {dx}"
    assert dy == 0, f"Expect dy to be 0, but got {dy}"

def test_axis_y():
    origin = [1, 2]
    destination = [3, 4]
    axis = 'y'
    dx, dy = _parse_move(origin, destination, axis)
    assert dx == 0, f"Expect dx to be 0, but got {dx}"
    assert dy == 2, f"Expect dy to be 2, but got {dy}"

def test_normal_case():
    origin = [1, 2]
    destination = [4, 6]
    axis = 'z'  
    dx, dy = _parse_move(origin, destination, axis)
    assert dx == 3, f"Expect dx to be 3, but got {dx}"
    assert dy == 4, f"Expect dy to be 4, but got {dy}"

def test_edge_case_origin_zero():
    origin = [0, 0]
    destination = [5, 7]
    axis = 'x'
    dx, dy = _parse_move(origin, destination, axis)
    assert dx == 5, f"Expect dx to be 5, but got {dx}"
    assert dy == 0, f"Expect dy to be 0, but got {dy}"

def test_edge_case_origin_negative():
    origin = [-1, -2]
    destination = [3, 4]
    axis = 'y'
    dx, dy = _parse_move(origin, destination, axis)
    assert dx == 0, f"Expect dx to be 0, but got {dx}"
    assert dy == 6, f"Expect dy to be 6, but got {dy}"

if __name__ == '__main__':
    pytest.main()

