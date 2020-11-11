#!/bin/sh

[ ! -d gdslib ] && git clone https://github.com/gdsfactory/gdslib.git

pip install -r requirements.txt --upgrade
pip install -r requirements_dev.txt --upgrade
pre-commit install
pf install
