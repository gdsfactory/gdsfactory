REM Windows installation script.
@echo off
[ ! -d gdslib ] && git clone https://github.com/gdsfactory/gdslib.git

conda install -c conda-forge gdspy
pip install -r requirements.txt --upgrade
pip install -r requirements_dev.txt --upgrade
python install_klive.py
python install_gdsdiff.py
python install_generic_tech.py
pip install pre-commit
pre-commit install
