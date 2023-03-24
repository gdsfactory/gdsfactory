# Workflow

You'll need 2 windows:

1. A text editor or IDE (Visual Studio Code, Pycharm, Spyder, neovim, Atom, Jupyterlab ...)
2. Klayout to Visualize the GDS files.

`Component.show()` will stream the GDS to KLayout so KLayout needs to be open.
Make sure you also run `gf install klayout-integration` from the terminal to install the `gdsfactory` to `klayout` interface.


## 1. Standard Python flow

1. You write your PCells in python.
2. You execute the python code.
3. You visualize the GDS Layout in Klayout.

![windows](https://i.imgur.com/ZHEAotn.png)


## 2. File-watcher flow

For building large components can use a file-watcher and see your updates in KLayout.

1. You execute the file watcher `gf watch FolderName` or in the current working directory `gf watch`
2. The file-watcher re-runs any python file `.py` or YAML `.pic.yaml`.
3. Thanks to the `cell` cache you can see your new component changes live updating the layout in Klayout.

![filewatcher](https://i.imgur.com/DNWgVRp.png)


## 3. Jupyter Notebook based flow

![notebooks](https://i.imgur.com/jORMG3V.png)
