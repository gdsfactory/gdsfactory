import pathlib
import pickle
from gdsfactory.simulation.gmeep import write_sparameters_meep

from gdsfactory.read import import_gds
from gdsfactory.technology import LayerStack

if __name__ == "__main__":
    with open("test/write_sparameters_meep_mpi.pkl", "rb") as inp:
        parameters_dict = pickle.load(inp)

    component = import_gds("test/write_sparameters_meep_mpi.gds", read_metadata=True)
    filepath_json = pathlib.Path("test/write_sparameters_meep_mpi.json")
    layer_stack = LayerStack.parse_raw(filepath_json.read_text())
    write_sparameters_meep(
        component=component,
        overwrite=True,
        layer_stack=layer_stack,
        filepath="/home/simbil/Github/ECE559/gdsfactory/gdslib/sp/straight_length2_add_pa_1ed425ca_969d5700.npz",
        ymargin=parameters_dict["ymargin"],
        resolution=parameters_dict["resolution"],
        is_3d=parameters_dict["is_3d"],
        wavelength_points=parameters_dict["wavelength_points"],
    )
