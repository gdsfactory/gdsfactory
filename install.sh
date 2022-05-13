#!/bin/sh

pip install -r requirements_dev.txt
pip install -e .
pre-commit install
gf tool install
