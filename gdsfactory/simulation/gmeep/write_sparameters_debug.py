import meep as mp
import numpy as np
import pandas as pd

import gdsfactory as gf
from gdsfactory.simulation.gmeep.get_simulation import get_simulation

if __name__ == "__main__":
    c = gf.components.straight(length=2)
    sim_dict = get_simulation(component=c)
    sim = sim_dict["sim"]
    monitors = sim_dict["monitors"]
    freqs = sim_dict["freqs"]
    field_monitor_point = sim_dict["field_monitor_point"]
    wavelengths = 1 / freqs
    filepath = "sp.csv"
    sources = sim_dict["sources"]
    source = sources[0]
    source_power = source.eig_power(1 / 1.55)
    port_source_name = sim_dict["port_source_name"]

    # sim.plot2D()
    # plt.show()

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            dt=50, c=mp.Ez, pt=field_monitor_point, decay_by=1e-9
        )
    )

    # call this function every 50 time spes
    # look at simulation and measure component that we want to measure (Ez component)
    # when field_monitor_point decays below a certain 1e-9 field threshold

    # Calculate mode overlaps
    nports = len(monitors)
    r = dict(wavelengths=wavelengths)
    S = np.zeros((len(freqs), nports, nports))
    a = {}
    b = {}

    # Parse out the overlaps
    for port_name, monitor in monitors.items():
        m_results = np.abs(sim.get_eigenmode_coefficients(monitor, [1]).alpha)
        a[port_name] = m_results[:, :, 0]  # forward wave
        b[port_name] = m_results[:, :, 1]  # backward wave

    source_fields = np.squeeze(a[port_source_name])
    for i, port_name_i in enumerate(monitors.keys()):
        monitor = monitors[port_name_i]
        r[f"s{i+1}"] = sim.get_eigenmode_coefficients(monitor, [1]).alpha[0, :, 0]

        # for j, port_name_j in enumerate(monitors.keys()):
        #     if port_name_j == port_name_i:
        #         S[:, i, j] = np.squeeze(b[port_name_j]) / source_fields
        #     else:
        #         S[:, i, j] = np.squeeze(a[port_name_j]) / source_fields

        # S[:, i, j] = np.squeeze(a[port_name_j] / b[port_name_i])
        # S[:, j, i] = np.squeeze(a[port_name_i] / b[port_name_j])

    # for port_name in monitor.keys():
    #     a1 = m1_results[:, :, 0]  # forward wave
    #     b1 = m1_results[:, :, 1]  # backward wave
    #     a2 = m2_results[:, :, 0]  # forward wave
    #     # b2 = m2_results[:, :, 1]  # backward wave

    #     # Calculate the actual scattering parameters from the overlaps
    #     s11 = np.squeeze(b1 / a1)
    #     s12 = np.squeeze(a2 / a1)

    keys = [key for key in r.keys() if key.startswith("s")]
    s = {f"{key}a": list(np.unwrap(np.angle(r[key].flatten()))) for key in keys}
    s.update({f"{key}m": list(np.abs(r[key].flatten())) for key in keys})
    s.update(wavelengths=wavelengths)
    s.update(freqs=freqs)
    df = pd.DataFrame(s)

    # df = df.set_index(df.wavelength)
    # df.to_csv(filepath, index=False)
    # s11 = S[:, 0, 0]
    # s12 = S[:, 0, 1]
    # s11 = np.abs(s11)
    # s12 = np.abs(s12)
    # s11 = 10 * np.log10(s11)
    # s12 = 10 * np.log10(s12)
    # plt.plot(wavelengths, s11, label="s11")
    # plt.plot(wavelengths, s12, label="s12")
    # plt.legend()

    monitor_out = monitors[2]
    source_power = np.abs(source_fields) ** 2
    transmission = (
        np.abs(sim.get_eigenmode_coefficients(monitor_out, [1]).alpha[0, :, 0]) ** 2
        / source_power
    )
    print(f"transmission: {transmission}")

    monitor_input = monitors[1]
    reflection = (
        np.abs(sim.get_eigenmode_coefficients(monitor_input, [1]).alpha[0, :, 1]) ** 2
        / source_power
    )
    print(f"Reflection: {reflection}")

    # sim.plot2D()
    # plt.show()
    # plt.show()
