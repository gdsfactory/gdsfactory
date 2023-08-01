import pathlib

import pandas as pd
import pytest
import gdsfactory as gf
from gdsfactory.plugins.pathlength_analysis import report_pathlengths

primitive_components = ["straight", "bend_euler", "bend_circular"]
supported_cross_sections = ["rib", "strip"]

_this_dir = pathlib.Path(__file__).parent

results_dir = _this_dir / "test_pathlength_reporting"
if not results_dir.is_dir():
    results_dir.mkdir()


@pytest.mark.parametrize("component_name", primitive_components)
def test_primitives_have_route_info(component_name):
    c = gf.get_component(component_name)
    assert "route_info" in c.info
    assert "length" in c.info
    assert c.info["length"] > 0
    assert c.info["route_info"]["length"] == c.info["length"]


@pytest.mark.parametrize("cross_section", supported_cross_sections)
def test_pathlength_extraction(cross_section):
    component_name = f"{test_pathlength_extraction.__name__}_{cross_section}"
    c = gf.Component(component_name)
    lengths = [10, 20, 45, 30]
    expected_total_length = sum(lengths)
    insts = []
    for i, length in enumerate(lengths):
        inst = c << gf.get_component(
            "straight", cross_section=cross_section, length=length
        )
        inst.name = f"s{i}"
        if insts:
            inst.connect("o1", insts[-1].ports["o2"])
        insts.append(inst)
    report_pathlengths(c, result_dir=results_dir)
    data = pd.read_csv(results_dir / f"{component_name}.pathlengths.csv")
    assert data.shape[0] == 1
    assert data["length"][0] == expected_total_length
    assert data[f"{cross_section.lower()}_length"][0] == expected_total_length
    src_inst = data["src_inst"][0]
    dst_int = data["dst_inst"][0]
    assert {src_inst, dst_int} == {"s0", f"s{len(lengths) - 1}"}
    src_port = data["src_port"][0]
    dst_port = data["dst_port"][0]
    assert {src_port, dst_port} == {"o1", "o2"}


def test_multi_path_extraction():
    component_name = f"{test_multi_path_extraction.__name__}"
    c = gf.Component(component_name)
    lengths1 = [10, 20, 45, 30]
    lengths2 = [50, 300]
    expected_total_length1 = sum(lengths1)
    expected_total_length2 = sum(lengths2)
    insts = []
    for i, length in enumerate(lengths1):
        inst = c << gf.get_component("straight", cross_section="strip", length=length)
        inst.name = f"s1-{i}"
        if insts:
            inst.connect("o1", insts[-1].ports["o2"])
        insts.append(inst)

    insts = []
    for i, length in enumerate(lengths2):
        inst = c << gf.get_component("straight", cross_section="strip", length=length)
        inst.movey(-100)
        inst.name = f"s2-{i}"
        if insts:
            inst.connect("o1", insts[-1].ports["o2"])
        insts.append(inst)
    report_pathlengths(c, result_dir=results_dir)
    data = pd.read_csv(results_dir / f"{component_name}.pathlengths.csv")
    assert data.shape[0] == 2
    extracted_lengths = data["length"].to_list()
    assert set(extracted_lengths) == {expected_total_length1, expected_total_length2}


@gf.cell
def pathlength_test_subckt(lengths, cross_section: str = "strip"):
    c1 = gf.Component()

    insts = []
    for i, length in enumerate(lengths):
        inst = c1 << gf.get_component(
            "straight", cross_section=cross_section, length=length
        )
        inst.name = f"s{i}"
        if insts:
            inst.connect("o1", insts[-1].ports["o2"])
        insts.append(inst)
    c1.add_port("o1", port=insts[0].ports["o1"])
    c1.add_port("o2", port=insts[-1].ports["o2"])
    return c1


def test_hierarchical_pathlength_extraction():
    component_name = f"{test_hierarchical_pathlength_extraction.__name__}"
    cross_section = "strip"
    c = gf.Component(component_name)
    lengths = [10, 20, 45, 30]
    expected_total_length_c1 = sum(lengths)
    c1 = pathlength_test_subckt(lengths, cross_section)
    istart = c.add_ref(c1, "istart")
    imid = c.add_ref(c1, "imid")
    iend = c.add_ref(
        gf.get_component("straight", cross_section=cross_section, length=100), "iend"
    )
    imid.connect("o1", istart.ports["o2"])
    iend.connect("o1", imid.ports["o2"])

    expected_total_length = 2 * expected_total_length_c1 + 100

    report_pathlengths(c, result_dir=results_dir)
    data = pd.read_csv(results_dir / f"{component_name}.pathlengths.csv")
    assert data.shape[0] == 1
    assert data["length"][0] == expected_total_length
    assert data[f"{cross_section.lower()}_length"][0] == expected_total_length
    src_inst = data["src_inst"][0]
    dst_int = data["dst_inst"][0]
    assert {src_inst, dst_int} == {"istart", "iend"}
    src_port = data["src_port"][0]
    dst_port = data["dst_port"][0]
    assert {src_port, dst_port} == {"o1", "o2"}


def test_transformed_hierarchical_pathlength_extraction():
    component_name = f"{test_transformed_hierarchical_pathlength_extraction.__name__}"
    cross_section = "strip"
    c = gf.Component(component_name)
    lengths = [10, 20, 45, 30]
    expected_total_length_c1 = sum(lengths)
    c1 = pathlength_test_subckt(lengths, cross_section)
    istart = c.add_ref(c1, "istart")
    imid = c.add_ref(c1, "imid")
    iend = c.add_ref(
        gf.get_component("straight", cross_section=cross_section, length=100), "iend"
    )
    istart = istart.rotate(37)
    imid.connect("o1", istart.ports["o2"])
    iend.connect("o1", imid.ports["o2"])

    expected_total_length = 2 * expected_total_length_c1 + 100
    report_pathlengths(c, result_dir=results_dir)
    data = pd.read_csv(results_dir / f"{component_name}.pathlengths.csv")
    assert data.shape[0] == 1
    assert data["length"][0] == expected_total_length
    assert data[f"{cross_section.lower()}_length"][0] == expected_total_length
    src_inst = data["src_inst"][0]
    dst_int = data["dst_inst"][0]
    assert {src_inst, dst_int} == {"istart", "iend"}
    src_port = data["src_port"][0]
    dst_port = data["dst_port"][0]
    assert {src_port, dst_port} == {"o1", "o2"}
