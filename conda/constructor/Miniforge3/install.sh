#!/bin/bash

source "$PREFIX/etc/profile.d/conda.sh"
conda activate "$PREFIX"

python -m pip install sklearn gdsfactory[full,dev,gmsh,tidy3d,devsim,meow,sax,ray,database,femwell,kfactory]==6.106.0

# conda install -c conda-forge slepc4py=*=complex* -y

# if [[ $(uname -s) == Linux ]]; then
#     if [[ ${INSTALLER_PLAT} != linux-* ]]; then
#         mamba install -c flaport klayout -y
#         mamba install -c flaport klayout-gui -y
#     fi
# else  # macOS
#   echo 'finished'
# fi
