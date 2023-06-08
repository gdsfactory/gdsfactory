#!/usr/bin/env bash

set -ex

conda install posix --yes
source scripts/build.sh
#source scripts/test.sh
set PIP_FIND_LINKS="https://whls.blob.core.windows.net/unstable/index.html"
pip install sax[nojax]
make full
pip install .
