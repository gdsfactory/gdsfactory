import pytest
from gdsfactory.functions import snap_angle

def test_snap_angle_zero():
    assert snap_angle(0) == 0

def test_snap_angle_positive_45():
    assert snap_angle(45) == 90

def test_snap_angle_negative_45():
    assert snap_angle(-45) == 0

def test_snap_angle_positive_135():
    assert snap_angle(135) == 180

def test_snap_angle_positive_225():
    assert snap_angle(225) == 270

def test_snap_angle_positive_315():
    assert snap_angle(315) == 0

def test_snap_angle_negative_135():
    assert snap_angle(-135) == 180

def test_snap_angle_negative_225():
    assert snap_angle(-225) == 270

def test_snap_angle_negative_315():
    assert snap_angle(-315) == 0

def test_snap_angle_wrap_around_370():
    assert snap_angle(370) == 0

def test_snap_angle_wrap_around_negative_370():
    assert snap_angle(-370) == 270

def test_snap_angle_edge_case_44_point_9999():
    assert snap_angle(44.9999) == 0

def test_snap_angle_edge_case_minus_44_point_9999():
    assert snap_angle(-44.9999) == 0

def test_snap_angle_edge_case_134_point_9999():
    assert snap_angle(134.9999) == 90

def test_snap_angle_edge_case_224_point_9999():
    assert snap_angle(224.9999) == 180

def test_snap_angle_edge_case_314_point_9999():
    assert snap_angle(314.9999) == 270

def test_snap_angle_close_to_zero():
    assert snap_angle(0.00001) == 0
    assert snap_angle(-0.00001) == 0

if __name__ == '__main__':
    pytest.main()

