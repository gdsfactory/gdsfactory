"""GDS regression test. Adapted from lytest.
"""
import pathlib

from lytest.kdb_xor import GeometryDifference, run_xor

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
    # we store the container with a different name from original component
    filename = (
        f"{component.function_name}_{component.name}.gds"
        if hasattr(component, "function_name")
        and component.name != component.function_name
        else f"{component.name}.gds"
    )

    ref_file = cwd / "gds_ref" / filename
    run_file = cwd / "gds_run" / filename
    diff_file = cwd / "gds_diff" / filename

    component.write_gds(gdspath=run_file)

    if not ref_file.exists():
        print(f"Creating GDS reference for {component.name} in {ref_file}")
        component.write_gds(gdspath=ref_file)
    try:
        run_xor(str(ref_file), str(run_file), tolerance=1, verbose=False)
    except GeometryDifference:
        diff = gdsdiff(ref_file, run_file, name=filename.split(".")[0])
        diff.write_gds(diff_file)
        diff.show()
        print(
            "\n"
            + f"`{filename}` changed from reference {ref_file}\n"
            + "You can check the differences in Klayout GUI\n"
            # + "If you want to save the current GDS as the new reference, type:\n"
            # f"rm {ref_file}"
        )

        try:
            val = input(
                "Would you like to save current GDS as the new reference? [y/n] "
            )
            if val.upper().startswith("Y"):
                print(f"rm {ref_file}")
                ref_file.unlink()
        except OSError:
            raise GeometryDifference(
                "\n"
                + f"`{filename}` changed from reference {ref_file}\n"
                + "To step over each error you can run `pytest -s`\n"
                + "So you can check the differences in Klayout GUI\n"
            )
