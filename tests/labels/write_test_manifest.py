import pandas as pd

import gdsfactory as gf
from gdsfactory.samples.sample_reticle import sample_reticle


def test_write_test_manifest():
    c = sample_reticle()
    gdspath = c.write_gds()
    csvpath = gdspath.with_suffix(".csv")
    gf.labels.write_test_manifest(component=c, csvpath=csvpath)
    df = pd.read_csv(csvpath)
    assert len(df) == 27


if __name__ == "__main__":
    c = sample_reticle()
    gdspath = c.write_gds()
    csvpath = gdspath.with_suffix(".csv")
    gf.labels.write_test_manifest(component=c, csvpath=csvpath)
    df = pd.read_csv(csvpath)
    print(len(df))
