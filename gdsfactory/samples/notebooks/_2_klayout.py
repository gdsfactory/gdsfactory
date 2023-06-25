# # KLayout integration
#
# [Klayout](https://www.klayout.de/build.html) is an open source layout viewer and editor. In gdsfactory code driven flow you define your components, circuits and reticles in python or YAML code.
#
# For rapid iteration, gdsfactory comes with a Klayout macro `klive` that runs inside klayout, so that when you run `component.show()` in python, it displays your GDS in Klayout.
#
# ![workflow](https://i.imgur.com/yquTcM7.png)

# You can install the klayout klive plugin to be able to see live updates on your GDS files:

# ![KLayout package](https://i.imgur.com/IZWH6U0.png)

# You can install the klayout generic pdk (layermap and DRC) in 2 ways:
#
# 1. from the terminal by typing `gf install klayout-genericpdk` after installing gdsfactory `pip install gdsfactory`
# 2. using KLayout package manager (see image below), Tools --> Manage Packages
#
# ![KLayout package](https://i.imgur.com/AkfcCms.png)

# What does the klayout generic tech installs?
#
# - generic layermap: for the generic gdsfactory technology.
# - generic DRC: for generic gdsfactory technology
