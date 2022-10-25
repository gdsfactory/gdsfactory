"""GDS regression test. Adapted from lytest.

TODO: adapt it into pytest_regressions

from pytest_regressions.file_regression import FileRegressionFixture

class GdsRegressionFixture(FileRegressionFixture):
    def check(self,
        contents,
        extension=".gds",
        basename=None,
        fullpath=None,
        binary=False,
        obtained_filename=None,
        check_fn=None,
            ):
        try:
            difftest(c)

"""
import filecmp
import pathlib
import shutil
from typing import Optional

from gdsfactory.component import Component
from gdsfactory.config import CONFIG, logger
from gdsfactory.gdsdiff.gdsdiff import gdsdiff


def difftest(
    component: Component,
    test_name: Optional[str] = None,
    xor: bool = False,
    dirpath: pathlib.Path = CONFIG["gdsdiff"],
) -> None:
    """Avoids GDS regressions tests on the GeometryDifference.

    If files are the same it returns None. If files are different runs XOR
    between new component and the GDS reference stored in dirpath and
    raises GeometryDifference if there are differences and show differences in klayout.

    If it runs for the fist time it just stores the GDS reference.


    Args:
        component: to test if it has changed.
        test_name: used to store the GDS file.
        xor: runs xor if there is difference.
        dirpath: defaults to cwd refers to where the test is being invoked.

    """
    from lytest.kdb_xor import GeometryDifference, run_xor

    # containers function_name is different from component.name
    # we store the container with a different name from original component
    test_name = test_name or (
        f"{component.function_name}_{component.name}"
        if hasattr(component, "function_name")
        and component.name != component.function_name
        else f"{component.name}"
    )
    filename = f"{test_name}.gds"
    ref_file = dirpath / "gds_ref" / filename
    run_file = dirpath / "gds_run" / filename
    diff_file = dirpath / "gds_diff" / filename

    component.write_gds(gdspath=run_file)

    if not ref_file.exists():
        component.write_gds(gdspath=ref_file)
        raise AssertionError(
            f"Reference GDS file for {test_name!r} not found. Writing to {ref_file!r}"
        )

    if filecmp.cmp(ref_file, run_file, shallow=False):
        return

    # component_reference = gf.import_gds(ref_file)
    # if component.hash_geometry() == component_reference.hash_geometry():
    #     print("same hash")
    #     return

    try:
        run_xor(str(ref_file), str(run_file), tolerance=1, verbose=False)
    except GeometryDifference as error:
        logger.error(error)
        diff = gdsdiff(ref_file, run_file, name=test_name, xor=xor)
        diff.write_gds(diff_file)
        diff.show(show_ports=False)
        print(
            f"\ngds_run {filename!r} changed from gds_ref {str(ref_file)!r}\n"
            "You can check the differences in Klayout GUI or run XOR with\n"
            f"gf gds diff --xor {ref_file} {run_file}\n"
        )

        try:
            val = input(
                "Would you like to save current GDS as the new reference? [Y/n] "
            )
            if val.upper().startswith("N"):
                raise
            logger.info(f"deleting file {str(ref_file)!r}")
            ref_file.unlink()
            shutil.copy(run_file, ref_file)
            raise
        except OSError as exc:
            raise GeometryDifference(
                "\n"
                f"{filename!r} changed from reference {str(ref_file)!r}\n"
                "To step over each error you can run `pytest -s`\n"
                "So you can check the differences in Klayout GUI\n"
            ) from exc


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.straight()
    difftest(c)
    # test_component(c, None, None)
