#!/bin/bash

mamba install pymeep=*=mpi_mpich_* -y

pip install gdsfactory[tidy3d]
pip install lytest simphony sax sklearn
pip install jax jaxlib


if [[ $(uname -s) == Linux ]]; then
    if [[ ${INSTALLER_PLAT} != linux-* ]]; then
        mamba install -c flaport klayout
        mamba install -c flaport klayout-gui
    fi
else  # macOS
    if [[ ${INSTALLER_PLAT} != osx-* ]]; then
        exit 1
    fi
fi
