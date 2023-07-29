#!/bin/bash

source "$PREFIX/etc/profile.d/conda.sh"
conda activate "$PREFIX"

python -m pip install gdsfactory[full,sax]==6.115.0
# python -m pip install sklearn gdsfactory[full,dev,gmsh,tidy3d,meow,sax,ray,database,femwell,kfactory]==6.115.0

# conda install -c conda-forge slepc4py=*=complex* -y

# if [[ $(uname -s) == Linux ]]; then
#     if [[ ${INSTALLER_PLAT} != linux-* ]]; then
#         mamba install -c flaport klayout -y
#         mamba install -c flaport klayout-gui -y
#     fi
# else  # macOS
#   echo 'finished'
# fi
