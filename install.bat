REM Windows installation script.
@echo off

git clone https://github.com/gdsfactory/gdslib.git -b data
conda install -c conda-forge gdspy
pip install -r requirements.txt --upgrade
pip install -r requirements_full.txt --upgrade
pip install -r requirements_dev.txt --upgrade
pre-commit install
gf tool install
