REM Windows installation script.
@echo off

conda install -c conda-forge gdspy
pip install -r requirements.txt --upgrade
pip install -r requirements_dev.txt --upgrade
pre-commit install
pf install
