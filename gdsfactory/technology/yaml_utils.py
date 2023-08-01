import yaml
from pydantic.color import Color

from gdsfactory.technology.color_utils import ensure_six_digit_hex_color


def add_color_yaml_presenter(prefer_named_color: bool = True) -> None:
    def _color_presenter(dumper: yaml.Dumper, data: Color) -> yaml.ScalarNode:
        data = data.as_named(fallback=True) if prefer_named_color else data.as_hex()
        return dumper.represent_scalar(
            "tag:yaml.org,2002:str", ensure_six_digit_hex_color(data), style='"'
        )

    yaml.add_representer(Color, _color_presenter)
    yaml.representer.SafeRepresenter.add_representer(Color, _color_presenter)


def add_tuple_yaml_presenter() -> None:
    def _tuple_presenter(dumper: yaml.Dumper, data: tuple) -> yaml.SequenceNode:
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    yaml.add_representer(tuple, _tuple_presenter)
    yaml.representer.SafeRepresenter.add_representer(tuple, _tuple_presenter)


def add_multiline_str_yaml_presenter() -> None:
    def _str_presenter(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
        if "\n" in data:  # check for multiline string
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    yaml.add_representer(str, _str_presenter)
    yaml.representer.SafeRepresenter.add_representer(str, _str_presenter)
