import gdsfactory as gf
import gdsfactory.simulation.gtidy3d as gt
from gdsfactory.config import CONFIG
from gdsfactory.simulation.gtidy3d.get_results import get_results


def test_results_local(data_regression) -> None:

    component = gf.components.straight(length=3)
    sim = gt.get_simulation(component=component)

    # dirpath = pathlib.Path(__file__).parent
    dirpath = CONFIG["sparameters"]
    r = get_results(sim=sim, dirpath=dirpath).result()
    data_regression.check(r.monitor_data)


if __name__ == "__main__":
    test_results_local(None)
