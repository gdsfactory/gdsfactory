"""GDS regression test."""
import pathlib

from lytest.kdb_xor import GeometryDifference, run_xor

import pp
from pp.component import Component
from pp.gdsdiff.gdsdiff import gdsdiff

cwd = pathlib.Path.cwd()


def difftest(component: Component) -> None:
    """Avoids GDS regressions tests on the GeometryDifference.
    Runs an XOR over a component and makes boolean comparison with a GDS reference.
    If it runs for the fist time it just stores the GDS reference.
    raises GeometryDifference if there are differences and show differences in klayout.
    """

    # containers function_name is different from component.name
    filename = (
        f"{component.function_name}_{component.name}.gds"
        if hasattr(component, "function_name")
        and component.name != component.function_name
        else f"{component.name}.gds"
    )

    ref_file = cwd / "gds_ref" / filename
    run_file = cwd / "gds_run" / filename
    diff_file = cwd / "gds_diff" / filename

    pp.write_gds(component, gdspath=run_file)

    if not ref_file.exists():
        print(f"Creating GDS reference for {component.name} in {ref_file}")
        pp.write_gds(component, gdspath=ref_file)
    try:
        run_xor(str(ref_file), str(run_file), tolerance=1, verbose=False)
    except GeometryDifference:
        diff = gdsdiff(ref_file, run_file, name=filename)
        pp.write_gds(diff, diff_file)
        pp.show(diff)
        raise
