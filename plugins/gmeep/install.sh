#!/bin/sh

conda install -c conda-forge pymeep
pip install -r requirements.txt
pip install -r requirements_dev.txt
pre-commit install
