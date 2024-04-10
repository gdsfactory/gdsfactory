import gdsfactory as gf


def test_info():
    c = gf.c.straight()
    c.info["test"] = "test"
    assert c.info["test"] == "test"
    c.info["length"] = 10
    c.info["d"] = dict(a=1, b=2)
    c.info["c"] = None
