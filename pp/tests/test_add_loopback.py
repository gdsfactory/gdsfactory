from pp.add_loopback import straight_with_loopback
from pp.component import Component
from pp.difftest import difftest


def test_add_loopback() -> Component:
    c = straight_with_loopback()
    difftest(c)
    return c


if __name__ == "__main__":
    c = test_add_loopback()
    c.show()
