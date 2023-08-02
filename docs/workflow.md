# Workflow

You'll need 2 windows:

1. A text editor or IDE (Visual Studio Code, Pycharm, Spyder, neovim, Atom, Jupyterlab ...)
2. Klayout to Visualize the GDS files. With the `klive` klayout package for showing GDS files when you run. Tools -> Manage Packages `Component.show()`

## 1. Standard Python flow

1. You write your PCells in python.
2. You execute the python code.
3. You visualize the GDS Layout in Klayout.

![windows](https://i.imgur.com/ZHEAotn.png)

The standard python flow can leverage all machine learning tools such as Jupyter notebooks.

![notebooks](https://i.imgur.com/jORMG3V.png)

## 2. File-watcher flow

For building large components can use a file-watcher and see your updates in KLayout.

1. You execute the file watcher `gf watch FolderName` or in the current working directory `gf watch`
2. The file-watcher re-runs any python file `.py` or YAML `.pic.yaml`.
3. Thanks to the `cell` cache you can see your new component changes live updating the layout in Klayout.

![filewatcher](https://i.imgur.com/DNWgVRp.png)

## 3. Webapp flow

You can also use a webapp to:

- visualize your PDK components and change the Parametric Cell (PCell) settings.
- visualize the GDS layout of the last saved files in a particular directory.
- turn on a file watcher and watch changing files in specific folders.

To run the webapp you can just run `gf webapp` in the terminal. By default it will use the `generic_pdk` technology.

To use a different PDK you can use the `--pdk` flag. For example `gf webapp --pdk ubcpdk`.
