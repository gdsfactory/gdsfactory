import yaml
import yaml.representer as yaml_representer
from pydantic_extra_types.color import Color

from gdsfactory.technology.color_utils import ensure_six_digit_hex_color


class TechnologyDumper(yaml.SafeDumper):
    pass


def add_color_yaml_representer(prefer_named_color: bool = True) -> None:
    """Add a custom YAML presenter for Color objects."""

    def _color_presenter(
        dumper: yaml_representer.SafeRepresenter, data: Color
    ) -> yaml.Node:
        data_str = data.as_named(fallback=True) if prefer_named_color else data.as_hex()
        return dumper.represent_scalar(
            "tag:yaml.org,2002:str", ensure_six_digit_hex_color(data_str), style='"'
        )

    TechnologyDumper.add_representer(Color, _color_presenter)


def add_tuple_yaml_representer() -> None:
    """Add a custom YAML presenter for tuple objects."""
    # Register the presenter only if not already present.
    if tuple not in TechnologyDumper.yaml_representers:
        TechnologyDumper.add_representer(tuple, _tuple_presenter)


def add_multiline_str_yaml_representer() -> None:
    """Add a custom YAML presenter for multiline strings."""

    def _str_presenter(
        dumper: yaml_representer.SafeRepresenter, data: str
    ) -> yaml.Node:
        if "\n" in data:  # check for multiline string
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    TechnologyDumper.add_representer(str, _str_presenter)


# Define the tuple presenter only once at module load time, not every call.
def _tuple_presenter(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


add_color_yaml_representer()
add_tuple_yaml_representer()
add_multiline_str_yaml_representer()
