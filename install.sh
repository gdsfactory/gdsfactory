#!/bin/sh

pip install -e .[dev] --upgrade
pre-commit install
gf tool install
