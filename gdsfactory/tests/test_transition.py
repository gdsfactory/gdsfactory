import pytest
from phidl.path import transition

import gdsfactory


def test_transition():
    with pytest.raises(ValueError):

        path = gdsfactory.Path()
        path.append(gdsfactory.path.arc(radius=10, angle=90))
        path.append(gdsfactory.path.straight(length=10))
        path.append(gdsfactory.path.euler(radius=3, angle=-90))
        path.append(gdsfactory.path.straight(length=40))
        path.append(gdsfactory.path.arc(radius=8, angle=-45))
        path.append(gdsfactory.path.straight(length=10))
        path.append(gdsfactory.path.arc(radius=8, angle=45))
        path.append(gdsfactory.path.straight(length=10))

        X = gdsfactory.CrossSection()
        X.add(width=1, offset=0, layer=0)

        x2 = gdsfactory.CrossSection()
        x2.add(width=2, offset=0, layer=0)

        transition(X, x2)


if __name__ == "__main__":
    test_transition()
