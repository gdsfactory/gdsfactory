""" working with units

- assert units are in um or m
- assert units are in arbitrary dbu (database units)

"""
wm = 1e-9
wm_min = 0.9e-9  # 1nm
wm_max = 999e-9  # 1um

wum = wm * 1e6
wum_min = wm_min * 1e6
wum_max = wm_max * 1e6


def assert_um(w):
    if hasattr(w, "__iter__"):
        w = w[0]
    assert wum_min < w < wum_max, f"are you sure `{w}` is in um?"


def assert_m(w):
    if hasattr(w, "__iter__"):
        w = w[0]
    assert wm_min < w < wm_max, f"are you sure `{w}` is in m?"


def assert_dbu(value, dbu=1e-3):
    if hasattr(value, "__iter__"):
        value = value[0]
    assert (
        value > dbu
    ), f"make sure {value} is in um and is larger than the minimum database unit {dbu}"


if __name__ == "__main__":
    w = [1e-9, 2e-9]
    assert_m(w)
    w = [1.55]
    assert_m(w)
