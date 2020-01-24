from pp.add_grating_couplers import add_te
from pp.add_grating_couplers import add_tm

from pp.routing.connect_component import add_io_optical_te
from pp.routing.connect_component import add_io_optical_tm

from pp.add_termination import add_gratings_and_loop_back_te
from pp.add_termination import add_gratings_and_loop_back_tm

name2function = {
    "add_grating_couplers_te": add_te,
    "add_grating_couplers_tm": add_tm,
    "add_io_optical_te": add_io_optical_te,
    "add_io_optical_tm": add_io_optical_tm,
    "add_gratings_and_loop_back_te": add_gratings_and_loop_back_te,
    "add_gratings_and_loop_back_tm": add_gratings_and_loop_back_tm,
}


if __name__ == "__main__":
    print(name2function)
