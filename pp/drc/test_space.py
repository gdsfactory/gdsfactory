import pp
from pp.drc import check_space


def test_space_fail():
    space = 0.12
    min_space = 0.2
    c = pp.c.waveguide_array(spacing=space)

    print(check_space(c, min_space=min_space))
    assert check_space(c, min_space=min_space) == 3600000


def test_space_pass():
    space = 0.12
    min_space = 0.1
    c = pp.c.waveguide_array(spacing=space)

    print(check_space(c, min_space=min_space))
    assert check_space(c, min_space=min_space) == 0


if __name__ == "__main__":
    test_space_fail()
