from __future__ import annotations

import gdsfactory as gf
import gdsfactory.simulation.gtidy3d as gt
from gdsfactory.config import PATH
from gdsfactory.simulation.gtidy3d.get_results import get_results

# def test_results_run(data_regression) -> None:
#     """Run simulations and checks local results."""

#     component = gf.components.straight(length=3)
#     sim = gt.get_simulation(component=component, is_3d=False)

#     dirpath = PATH.sparameters
#     r = get_results(sim=sim, dirpath=dirpath, overwrite=True).result()

#     if data_regression:
#         data_regression.check(r.monitor_data)


if __name__ == "__main__":
    # test_results_run(None)

    component = gf.components.straight(length=3)
    sim = gt.get_simulation(component=component, is_3d=False)

    dirpath = PATH.sparameters
    r = get_results(sim=sim, dirpath=dirpath, overwrite=True).result()
