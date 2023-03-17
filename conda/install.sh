#!/bin/bash

conda install -c conda-forge slepc4py=*=complex* -y
pip install sklearn gdsfactory[full,gmsh,tidy3d,devsim,meow,sax,ray,database,femwell]==6.61.0
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
