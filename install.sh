#!/bin/sh

pip install -r requirements_full.txt
pip install -e .
pre-commit install
gf tool install
make plugins


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
