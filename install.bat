REM Windows installation script.
@echo off
[ ! -d gdslib ] && git clone https://github.com/gdsfactory/gdslib.git

conda install -c conda-forge gdspy
pip install -r requirements.txt --upgrade
pip install -r requirements_dev.txt --upgrade
pip install pre-commit
pre-commit install
pf install
