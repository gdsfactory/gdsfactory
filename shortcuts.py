import json
import sys

import menuinst

spyder_settings = """
{
    "menu_name": "Spyder",
    "menu_items":
        [
            {
                "script": "${PREFIX}/Scripts/spyder.exe",
                "scriptarguments": [],
                "name": "Spyder",
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
    "menu_name": "JupyterLab",
    "menu_items":
        [
            {
                "script": "${PREFIX}/Scripts/jupyter-lab.exe",
                "scriptarguments": [],
                "name": "JupyterLab",
                "workdir": "${PREFIX}",
                "icon": "${PREFIX}/Lib/site-packages/nbclassic/static/base/images/favicon.ico",
                "desktop": true,
                "quicklaunch": true
            }
        ]
}
"""


ananconda_navigator_settings = """
{
    "menu_name": "Anaconda Navigator",
    "menu_items":
        [
            {
                "script": "${PREFIX}/Scripts/anaconda-navigator.exe",
                "scriptarguments": [],
                "name": "Anaconda Navigator",
                "workdir": "${PREFIX}",
                "icon": "${MENU_DIR}/anaconda-navigator.ico",
                "desktop": true,
                "quicklaunch": true
            }
        ]
}
"""


def shortcuts(settings: str):
    with open(f"{sys.prefix}/settings.json", "w") as f:
        json.dump(json.loads(settings), f)
    menuinst.install("settings.json", prefix=f"{sys.argv[1]}")


if __name__ == "__main__":
    icons = [spyder_settings, jupyter_lab_settings, ananconda_navigator_settings]
    for icon in icons:
        shortcuts(icon)
