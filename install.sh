#!/bin/bash

mamba install pymeep=*=mpi_mpich_* -y

pip install jax jaxlib
pip install gdsfactory[tidy3d]
pip install gdsfactory[full] --upgrade


# if [[ $(uname -s) == Linux ]]; then
#     if [[ ${INSTALLER_PLAT} != linux-* ]]; then
#         mamba install -c flaport klayout -y
#         mamba install -c flaport klayout-gui -y
#     fi
# else  # macOS
#   echo 'finished'
# fi
