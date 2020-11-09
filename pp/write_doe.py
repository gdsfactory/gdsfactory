import json
from pp.name import get_component_name
from pp.components import component_factory
from pp.write_component import write_component
from pp.config import CONFIG
from pp.doe import get_settings_list
from pp.routing.add_fiber_array import add_fiber_array_te, add_fiber_array_tm

function_factory = dict(
    add_fiber_array_te=add_fiber_array_te, add_fiber_array_tm=add_fiber_array_tm
)


def write_doe_metadata(
    doe_name,
    cell_names,
    list_settings,
    doe_settings=None,
    doe_metadata_path=CONFIG["doe_directory"],
    **kwargs,
):
    """writes DOE metadata (markdown report, JSON dict)

    Args:
        doe_name
        cell_names: list of klayout <Cell> or pp <Component>
        list_settings: list of settings for the cell
        doe_settings: test and data_analysis_protocol
        cell_settings: list of cell settings

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
            for header, fields in zip(head, list_data_str):
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
    component_type,
    doe_name,
    do_permutations=True,
    list_settings=None,
    doe_settings=None,
    path=CONFIG["build_directory"],
    doe_metadata_path=CONFIG["doe_directory"],
    functions=None,
    function_factory=function_factory,
    component_factory=component_factory,
    **kwargs,
):
    """writes each device GDS, together with metadata for each device:
    Returns a list of gdspaths

    - .gds geometry for each component
    - .json metadata for each component
    - .ports if any for each component
    - report.md for DOE
    - doe_name.json with DOE metadata

    pp.write_component_doe("mmi1x2", width_mmi=[5, 10], length_mmi=9)

    Args:
        component_type: component_name_or_function
        doe_name: name of the DOE
        do_permutations: builds all permutations between the varying parameters
        list_settings: you can pass a list of settings or the variations in the kwargs
        doe_settings: shared settings for a DOE
        path: to store build artifacts
        functions: list of function names to apply to DOE
        function_factory: function names to functions dict
        **kwargs: Doe default settings or variations
    """

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
            component = function_factory[f](component)

        cell_names.append(component.name)
        cell_settings.append(settings)
        gdspath = path / f"{component.name}.gds"
        doe_gds_paths += [gdspath]
        write_component(component, gdspath)

    """ write DOE metadata (report + JSON) """
    write_doe_metadata(
        doe_name=doe_name,
        cell_names=cell_names,
        list_settings=list_settings,
        doe_settings=doe_settings,
        cell_settings=kwargs,
        doe_metadata_path=doe_metadata_path,
    )

    return doe_gds_paths


def get_markdown_table(do_permutations=True, **kwargs):
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
    for header, fields in zip(head, list_data_str):
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
        return "|".join(fmt_fields)

    def table_head_sep():
        return "|".join(["-" * n for n in field_sizes])

    t = []
    t.append(fmt_line(head))
    t.append(table_head_sep())
    for fields in list_data_str[0:]:
        t.append(fmt_line(fields))

    return t


def test_write_doe():
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
        functions=["add_fiber_array_tm"],
        doe_settings=dict(test="optical_tm"),
    )
    assert len(paths) == 4
    return paths[0]


if __name__ == "__main__":
    import pp

    path = test_write_doe()
    pp.show(path)
    # print(get_markdown_table(width_mmi=[5, 6]))
    # paths = write_doe(
    #     "mmi1x2", width_mmi=[5, 10], length_mmi=[20, 30], do_permutations=False
    # )
    # print(paths)
    # gdspaths = test_write_doe()
