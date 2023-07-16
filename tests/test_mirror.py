import gdsfactory as gf


def test_mirror() -> None:
    c1 = gf.components.pad()
    c1.mirror()


if __name__ == "__main__":
    test_mirror()
