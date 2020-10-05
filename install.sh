#!/bin/sh

[ ! -d gdslib ] && git clone https://github.com/gdsfactory/gdslib.git

pip install -r requirements.txt --upgrade
pip install -r requirements_dev.txt --upgrade
python install_klive.py
python install_gdsdiff.py
python install_generic_tech.py
pip install pre-commit
pre-commit install
