#!/bin/bash

pip install gdsfactory[tidy3d]
pip install gdsfactory[full] --upgrade
pip install gdsfactory==5.33.9
gf tool install

[ ! -d $HOME/gdsfactory ] && git clone https://github.com/gdsfactory/gdsfactory.git $HOME/gdsfactory

# if [[ $(uname -s) == Linux ]]; then
#     if [[ ${INSTALLER_PLAT} != linux-* ]]; then
#         mamba install -c flaport klayout -y
#         mamba install -c flaport klayout-gui -y
#     fi
# else  # macOS
#   echo 'finished'
# fi
