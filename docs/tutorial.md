# Layout Tutorial

We store the tutorial as python files with [jupytext](https://jupytext.readthedocs.io/en/latest/) tags.
Jupytext is a python package to convert jupyter notebook into text (python or markdown)

You can use [VSCode Jupytext extension](https://marketplace.visualstudio.com/items?itemName=congyiwu.vscode-jupytext) to open the notebooks from the python files.

<img src=https://raw.githubusercontent.com/notebookPowerTools/vscode-jupytext/main/images/main.gif>

If you really want to use notebooks (not recommended) the `Makefile` has some helper commands:

- `make notebooks`: makes the `ipynb` notebooks from the `py` python files.
- `make jupytext`: convert `ipynb` notebooks to python in case you modified the notebooks, so you can track changes.
