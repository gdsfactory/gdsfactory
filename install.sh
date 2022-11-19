#!/bin/bash

pip install gdsfactory[tidy3d]
pip install gdsfactory[full] --upgrade
pip install devsim
pip install gdsfactory==6.1.1
gf tool install

[ ! -d $HOME/Desktop/gdsfactory ] && git clone https://github.com/gdsfactory/gdsfactory.git $HOME/Desktop/gdsfactory

# if [[ $(uname -s) == Linux ]]; then
#     if [[ ${INSTALLER_PLAT} != linux-* ]]; then
#         mamba install -c flaport klayout -y
#         mamba install -c flaport klayout-gui -y
#     fi
# else  # macOS
#   echo 'finished'
# fi
