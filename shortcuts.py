import json
import sys

import menuinst


spyder_settings = """
{
    "menu_name": "gdsfactory Spyder",
    "menu_items":
        [
            {
                "script": "${PREFIX}/Scripts/spyder.exe",
                "scriptarguments": [],
                "name": "gdsfactory Spyder",
                "workdir": "${PREFIX}",
                "icon": "${MENU_DIR}/spyder.ico",
                "desktop": true,
                "quicklaunch": true
            }
        ]
}
"""


jupyter_lab_settings = """
{
    "menu_name": "gdsfactory JupyterLab",
    "menu_items":
        [
            {
                "script": "${PREFIX}/Scripts/jupyter-lab.exe",
                "scriptarguments": [],
                "name": "gdsfactory JupyterLab",
                "workdir": "${PREFIX}",
                "icon": "${PREFIX}/Lib/site-packages/nbclassic/static/base/images/favicon.ico",
                "desktop": true,
                "quicklaunch": true
            }
        ]
}
"""


anaconda_navigator_settings = """
{
    "menu_name": "gdsfactory Navigator",
    "menu_items":
        [
            {
                "script": "${PREFIX}/Scripts/anaconda-navigator.exe",
                "scriptarguments": [],
                "name": "gdsfactory Navigator",
                "workdir": "${PREFIX}",
                "icon": "${MENU_DIR}/anaconda-navigator.ico",
                "desktop": true,
                "quicklaunch": true
            }
        ]
}
"""


def shortcuts(settings: str):
    with open("settings.json", "w") as f:
        json.dump(json.loads(settings), f)
    menuinst.install("settings.json", prefix=sys.executable.replace("python.exe", ""))


if __name__ == "__main__":
    icons = [spyder_settings, jupyter_lab_settings, anaconda_navigator_settings]
    for icon in icons:
        shortcuts(icon)
