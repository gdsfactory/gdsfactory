# this code has been automatically generated from pp/sp/write_tests.py
import pp


def test_load_sparameters_waveguide(data_regression):
    c = pp.c.waveguide()
    sp = pp.sp.load(c)

    port_names = sp[0]
    f = list(sp[1])
    s = sp[2]

    lenf = s.shape[0]
    rows = s.shape[1]
    cols = s.shape[2]

    assert rows == cols == len(c.ports)
    assert len(port_names) == len(c.ports)
    data_regression.check(dict(port_names=port_names))
    assert lenf == len(f)


def test_load_sparameters_bend_circular(data_regression):
    c = pp.c.bend_circular()
    sp = pp.sp.load(c)

    port_names = sp[0]
    f = list(sp[1])
    s = sp[2]

    lenf = s.shape[0]
    rows = s.shape[1]
    cols = s.shape[2]

    assert rows == cols == len(c.ports)
    assert len(port_names) == len(c.ports)
    data_regression.check(dict(port_names=port_names))
    assert lenf == len(f)


def test_load_sparameters_coupler(data_regression):
    c = pp.c.coupler()
    sp = pp.sp.load(c)

    port_names = sp[0]
    f = list(sp[1])
    s = sp[2]

    lenf = s.shape[0]
    rows = s.shape[1]
    cols = s.shape[2]

    assert rows == cols == len(c.ports)
    assert len(port_names) == len(c.ports)
    data_regression.check(dict(port_names=port_names))
    assert lenf == len(f)


def test_load_sparameters_mmi1x2(data_regression):
    c = pp.c.mmi1x2()
    sp = pp.sp.load(c)

    port_names = sp[0]
    f = list(sp[1])
    s = sp[2]

    lenf = s.shape[0]
    rows = s.shape[1]
    cols = s.shape[2]

    assert rows == cols == len(c.ports)
    assert len(port_names) == len(c.ports)
    data_regression.check(dict(port_names=port_names))
    assert lenf == len(f)


def test_load_sparameters_mmi2x2(data_regression):
    c = pp.c.mmi2x2()
    sp = pp.sp.load(c)

    port_names = sp[0]
    f = list(sp[1])
    s = sp[2]

    lenf = s.shape[0]
    rows = s.shape[1]
    cols = s.shape[2]

    assert rows == cols == len(c.ports)
    assert len(port_names) == len(c.ports)
    data_regression.check(dict(port_names=port_names))
    assert lenf == len(f)
