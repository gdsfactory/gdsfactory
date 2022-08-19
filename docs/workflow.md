# Workflow

You'll need 2 windows:

1. A text editor or IDE (Visual Studio Code, Pycharm, Spyder, neovim, Atom, Jupyterlab ...)
2. Klayout to Visualize the GDS files.

`Component.show()` will stream the GDS to klayout so klayout needs to be open.
Make sure you also ran `gf tool install` from the terminal to install the `gdsfactory` to `klayout` interface.


## 1. Python driven flow

1. You write your Pcells in python.
2. You execute the python code.
3. You visualize the GDS Layout in Klayout.

![windows](https://i.imgur.com/ZHEAotn.png)


## 2. YAML driven flow

For building complex circuits and assemble your reticle DOEs (design of experiment) you can also use the YAML driven flow.

With a file watcher you can see your changes live in klayout.


1. You execute the file watcher `gf yaml watch FolderName`.
2. You write your component or circuit in YAML as Place and Auto-Route.
3. You visualize the GDS Layout in Klayout.

![yaml](https://i.imgur.com/h1ABhJ9.png)
