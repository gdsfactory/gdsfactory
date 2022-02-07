from jsondiff import diff
from omegaconf import OmegaConf

import gdsfactory as gf
import gdsfactory.simulation.gtidy3d as gt
from gdsfactory.simulation.gtidy3d.get_results import get_sim_hash


def test_simulation_hash():
    component = gf.components.straight(length=3)
    sim = gt.get_simulation(component=component)
    sim_hash = get_sim_hash(sim)
    # print(f"assert hash == {sim_hash!r}")
    assert sim_hash == "5cf1811250c66398ff88c62f509ae7aa", sim_hash


def test_simulation():
    """export sim in JSON, and then load it again"""
    component = gf.components.straight(length=3)
    sim = gt.get_simulation(component=component)
    # sim.to_file("sim_ref.yaml")
    sim.to_file("sim_run.yaml")

    dref = OmegaConf.load("sim_ref.yaml")
    drun = OmegaConf.load("sim_run.yaml")

    d = diff(dref, drun)
    assert len(d) == 0, d


if __name__ == "__main__":
    test_simulation()

    # component = gf.components.straight(length=3)
    # sim = gt.get_simulation(component=component)
    # # sim.to_file("sim_ref.yaml")
    # sim.to_file("sim_run.yaml")

    # dref = OmegaConf.load("sim_ref.yaml")
    # drun = OmegaConf.load("sim_run.yaml")

    # d = diff(dref, drun)
