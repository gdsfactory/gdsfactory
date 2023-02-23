# Layout Tutorial

The tutorial is stored as python files with jupytext tags.

You can use [VSCode Jupytext extension](https://marketplace.visualstudio.com/items?itemName=donjayamanne.vscode-jupytext) to open the notebooks from the python files.


If you really want to use notebooks (not recommended) the `Makefile` has some helper commands

`make notebooks`: makes the `ipynb` notebooks from the `py` python files.
`make jupytext`: convert `ipynb` notebooks to python in case you modified the notebooks, so you can track changes.

```{tableofcontents}
```
