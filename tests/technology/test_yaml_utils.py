from io import StringIO

from pydantic_extra_types.color import Color

from gdsfactory.technology.yaml_utils import (
    TechnologyDumper,
    add_color_yaml_representer,
)


def test_color_yaml_representer_named() -> None:
    add_color_yaml_representer(prefer_named_color=True)
    color = Color("red")
    stream = StringIO()
    dumper = TechnologyDumper(stream)
    dumper.open()
    dumper.represent(color)
    dumper.close()
    yaml_str = stream.getvalue()
    assert yaml_str.strip() == '"red"'


def test_color_yaml_representer_hex() -> None:
    add_color_yaml_representer(prefer_named_color=False)
    color = Color("#ff0000")
    stream = StringIO()
    dumper = TechnologyDumper(stream)
    dumper.open()
    dumper.represent(color)
    dumper.close()
    yaml_str = stream.getvalue()
    assert yaml_str.strip() == '"#ff0000"'


def test_tuple_yaml_representer() -> None:
    test_tuple = (1, 2, 3)
    stream = StringIO()
    dumper = TechnologyDumper(stream)
    dumper.open()
    dumper.represent(test_tuple)
    dumper.close()
    yaml_str = stream.getvalue()
    assert yaml_str.strip() == "[1, 2, 3]"


def test_multiline_str_yaml_representer() -> None:
    single_line = "Hello"
    multi_line = "Hello\nWorld"

    stream2 = StringIO()
    dumper2 = TechnologyDumper(stream2)
    dumper2.open()
    dumper2.represent(multi_line)
    dumper2.close()
    multi_line_yaml = stream2.getvalue()
    assert "|-" in multi_line_yaml
    assert "Hello\n  World" in multi_line_yaml

    stream1 = StringIO()
    dumper1 = TechnologyDumper(stream1)
    dumper1.open()
    dumper1.represent(single_line)
    dumper1.close()
    single_line_yaml = stream1.getvalue()
    assert "Hello" in single_line_yaml.strip()
