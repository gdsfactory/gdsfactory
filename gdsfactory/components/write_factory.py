import inspect
import pathlib

from gdsfactory import components
from gdsfactory.component import Component


def write_factory(module=components, filepath="write_factory.pyc"):
    """Write a file with all the component factories into a dict."""
    factory = [
        i
        for i in dir(module)
        if not i.startswith("_")
        and callable(getattr(components, i))
        and type(inspect.signature(getattr(components, i)).return_annotation)
        == type(Component)
    ]

    lines = [f"{i}={i}" for i in factory]
    script = "factory = dict(" + ",".join(lines) + ")"
    filepath = pathlib.Path(filepath)
    print(script)
    filepath.write_text(script)


if __name__ == "__main__":
    write_factory()
