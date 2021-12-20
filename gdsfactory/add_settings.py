import json

from gdsfactory.component import Component


def add_settings_from_label(component: Component) -> None:
    """Adds settings from label in JSON format in the GDS."""
    for label in component.labels:
        if label.text.startswith("settings="):
            d = json.loads(label.text[9:])
            component._settings_full = d.pop("settings", {})
            for k, v in d.items():
                setattr(component, k, v)
