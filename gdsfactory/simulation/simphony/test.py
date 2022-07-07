import matplotlib.pyplot as plt

from simphony.libraries import siepic, sipann
from simphony.simulators import MonteCarloSweepSimulator, SweepSimulator

# y_splitter = siepic.YBranch()
# wg_long = siepic.Waveguide(length=150e-6)
# wg_short = siepic.Waveguide(length=50e-6)
# y_recombiner = siepic.YBranch()

# # or connect components with components:
# # (when using components to make connections, their first unconnected pin will
# # be used to make the connection.)
# y_splitter["pin2"].connect(wg_long)

# # or any combination of the two:
# y_splitter["pin3"].connect(wg_short)
# # y_splitter.connect(wg_short["pin1"])

# # when making multiple connections, it is often simpler to use `multiconnect`
# # multiconnect accepts components, pins, and None
# # if None is passed in, the corresponding pin is skipped

dc = sipann.Standard(0.5e-6,0.22e-6,0.22e-6,10e-6,1.5e-6,2.5e-6)
wg = siepic.Waveguide(1e-6)
wg2 = siepic.Waveguide(1e-6)
dc[0].connect(wg)
dc[2].connect(wg2)
# y_recombiner.multiconnect(None, wg_short, wg_long)

simulator = MonteCarloSweepSimulator(1500e-9, 1600e-9)
simulator.multiconnect(wg, wg2)

results = simulator.simulate(runs=10)
for f, p in results:
  plt.plot(f, p)

f, p = results[0]
plt.plot(f, p, "k")
plt.title("MZI Monte Carlo")
plt.tight_layout()
plt.show()
