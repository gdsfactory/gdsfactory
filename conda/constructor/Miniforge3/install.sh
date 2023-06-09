#!/bin/bash

conda install -c conda-forge slepc4py=*=complex* -y
pip install sklearn gdsfactory[full,dev,gmsh,tidy3d,devsim,meow,sax,ray,database,femwell,kfactory]==6.103.6

# if [[ $(uname -s) == Linux ]]; then
#     if [[ ${INSTALLER_PLAT} != linux-* ]]; then
#         mamba install -c flaport klayout -y
#         mamba install -c flaport klayout-gui -y
#     fi
# else  # macOS
#   echo 'finished'
# fi
