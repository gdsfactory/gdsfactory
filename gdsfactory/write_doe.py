import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from gdsfactory.cell import get_component_name
from gdsfactory.component import Component
from gdsfactory.components import LIBRARY
from gdsfactory.config import CONFIG
from gdsfactory.doe import get_settings_list
from gdsfactory.tech import Library


def write_doe_metadata(
    doe_name: str,
    cell_names: List[str],
    list_settings: Union[List[Dict[str, Union[float, int]]], List[Dict[str, int]]],
    doe_settings: Optional[Dict[str, str]] = None,
    doe_metadata_path: Path = CONFIG["doe_directory"],
    **kwargs,
) -> None:
    """writes DOE metadata (markdown report, JSON dict)

    Args:
        doe_name
        cell_names: list of klayout <Cell> or gf.<Component>
        list_settings: list of settings for the cell
        doe_settings: test and data_analysis_protocol
        doe_metadata_path

    """

    doe_metadata_path.mkdir(parents=True, exist_ok=True)
    report_path = doe_metadata_path / (doe_name + ".md")
    doe_settings = doe_settings or {}
    json_path = report_path.with_suffix(".json")

    d = dict(
        type="doe",
        name=doe_name,
        cells={cell_name: {"name": cell_name} for cell_name in cell_names},
        settings=list_settings,
        doe_settings=doe_settings,
        **kwargs,
    )

    with open(json_path, "w+") as fw:
        fw.write(json.dumps(d, indent=2))

    with open(report_path, "w+") as fw:

        def w(line=""):
            fw.write(line + "\n")

        w("# {}".format(doe_name))
        w("- Number of devices: {}".format(len(list_settings)))
        w("- Settings")

        if len(kwargs) > 0:
            w(json.dumps(doe_settings, indent=2))

        # w(json.dumps(list_settings))
        if list_settings:

            w()
            EOL = ""

            # Convert table fields to strings
            head = [
                [str(field) for field in fields.keys()] for fields in list_settings
            ][0]
            list_data_str = [
                [str(field) for field in fields.values()] for fields in list_settings
            ]
            N = len(head)

            field_sizes = [0] * N

            # compute the max number of character in each field
            for _, fields in zip(head, list_data_str):
                field_sizes = [
                    max(field_sizes[i], len(fields[i]), len(head[i])) for i in range(N)
                ]

            field_sizes = [n + 2 for n in field_sizes]

            # Line formatting from fields
            def fmt_line(fields):
                fmt_fields = [
                    " " + fields[i] + " " * (field_sizes[i] - len(fields[i]) - 1)
                    for i in range(N)
                ]
                return "|".join(fmt_fields) + EOL

            def table_head_sep():
                return "|".join(["-" * n for n in field_sizes]) + EOL

            w(fmt_line(head))
            w(table_head_sep())
            for fields in list_data_str[0:]:
                w(fmt_line(fields))

            w()
            w("Cells: \n")
            for cell_name in cell_names:
                w(f"- {cell_name}")

            w()


def write_doe(
    component_type: str,
    doe_name: str,
    do_permutations: bool = True,
    list_settings: Optional[
        Union[List[Dict[str, Union[float, int]]], List[Dict[str, int]]]
    ] = None,
    doe_settings: Optional[Dict[str, str]] = None,
    path: Path = CONFIG["build_directory"],
    doe_metadata_path: Path = CONFIG["doe_directory"],
    functions: Optional[List[Callable[..., Component]]] = None,
    library: Library = LIBRARY,
    **kwargs,
) -> List[Path]:
    """writes each component GDS, together with metadata for each component:
    Returns a list of gdspaths

    - .gds geometry for each component
    - .json metadata for each component
    - .ports if any for each component
    - report.md for DOE
    - doe_name.json with DOE metadata

    gf.write_component_doe("mmi1x2", width_mmi=[5, 10], length_mmi=9)

    Args:
        component_type: component_name_or_function
        doe_name: name of the DOE
        do_permutations: builds all permutations between the varying parameters
        list_settings: you can pass a list of settings or the variations in the kwargs
        doe_settings: shared settings for a DOE
        path: to store build artifacts
        functions: list of function names to apply to DOE
        **kwargs: Doe default settings or variations
    """
    component_factory = library.factory

    component_type = (
        component_type.__name__ if callable(component_type) else component_type
    )

    functions = functions or []
    list_settings = list_settings or get_settings_list(
        do_permutations=do_permutations, **kwargs
    )

    assert isinstance(component_type, str), f"{component_type} not recognized"

    path.mkdir(parents=True, exist_ok=True)

    doe_gds_paths = []
    cell_names = []
    cell_settings = []

    for settings in list_settings:
        # print(settings)
        component_name = get_component_name(component_type, **settings)
        component_function = component_factory[component_type]
        component = component_function(name=component_name, **settings)
        if "test" in kwargs:
            component.test_protocol = kwargs.get("test")
        if "analysis" in kwargs:
            component.data_analysis_protocol = kwargs.get("analysis")
        for f in functions:
            component = f(component)

        cell_names.append(component.name)
        cell_settings.append(settings)
        gdspath = path / f"{component.name}.gds"
        doe_gds_paths += [gdspath]
        component.write_gds_with_metadata(gdspath)

    write_doe_metadata(
        doe_name=doe_name,
        cell_names=cell_names,
        list_settings=list_settings,
        doe_settings=doe_settings,
        cell_settings=kwargs,
        doe_metadata_path=doe_metadata_path,
    )

    return doe_gds_paths


def get_markdown_table(do_permutations=True, **kwargs) -> List[str]:
    """returns the markdown table for a parameter sweep"""
    list_settings = get_settings_list(do_permutations=do_permutations, **kwargs)
    # Convert table fields to strings
    head = [[str(field) for field in fields.keys()] for fields in list_settings][0]
    list_data_str = [
        [str(field) for field in fields.values()] for fields in list_settings
    ]
    N = len(head)

    field_sizes = [0] * N

    # compute the max number of character in each field
    for _, fields in zip(head, list_data_str):
        field_sizes = [
            max(field_sizes[i], len(fields[i]), len(head[i])) for i in range(N)
        ]

    field_sizes = [n + 2 for n in field_sizes]

    # Line formatting from fields
    def fmt_line(fields) -> str:
        fmt_fields = [
            " " + fields[i] + " " * (field_sizes[i] - len(fields[i]) - 1)
            for i in range(N)
        ]
        return "|".join(fmt_fields)

    def table_head_sep() -> str:
        return "|".join(["-" * n for n in field_sizes])

    t = []
    t.append(fmt_line(head))
    t.append(table_head_sep())
    for fields in list_data_str[0:]:
        t.append(fmt_line(fields))

    return t


def test_write_doe() -> Path:
    paths = write_doe(
        component_type="mmi1x2",
        doe_name="width_length",
        width_mmi=[5, 10],
        length_mmi=[20, 30],
        do_permutations=False,
    )
    assert len(paths) == 2
    paths = write_doe(
        component_type="mmi1x2",
        doe_name="width_length2",
        width_mmi=[5, 10],
        length_mmi=[20, 30],
        do_permutations=True,
        doe_settings=dict(test="optical_tm"),
    )
    assert len(paths) == 4
    return paths[0]


if __name__ == "__main__":
    import gdsfactory as gf

    path0 = test_write_doe()
    gf.show(path0)

    # print(get_markdown_table(width_mmi=[5, 6]))
    # paths = write_doe(
    #     "mmi1x2", width_mmi=[5, 10], length_mmi=[20, 30], do_permutations=False
    # )
    # print(paths)
    # gdspaths = test_write_doe()
