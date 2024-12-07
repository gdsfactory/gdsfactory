import pathlib
from inspect import Parameter, Signature, signature
from io import IOBase
from typing import IO, Any

import jinja2
import yaml
from kfactory import cell

from gdsfactory.component import Component
from gdsfactory.read.from_yaml import from_yaml
from gdsfactory.typings import ComponentFactory, RoutingStrategies

__all__ = ["cell_from_yaml_template"]

_YamlDefinition = str | IO[Any] | pathlib.Path


def split_default_settings_from_yaml(yaml_lines: list[str]) -> tuple[str, str]:
    """Separates out the 'default_settings' block from the rest of the file body.

    Note: 'default settings' MUST be at the TOP of the file.

    Args:
        yaml_lines: the lines of text in the yaml file.

    Returns:
        a tuple of (main file contents), (setting block), both as multi-line strings.
    """
    settings_lines = []
    other_lines = []
    # start reading all lines
    while yaml_lines:
        # pop lines until we find the default_settings block
        line = yaml_lines.pop(0)
        if line.startswith("default_settings"):
            settings_lines.append(line)
            # keep adding lines to settings until we find a new top-level block...
            # then we will add the rest of the lines to the main file block
            while yaml_lines:
                next_line = yaml_lines.pop(0)
                if next_line[0].isspace():
                    settings_lines.append(next_line)
                else:
                    other_lines.append(next_line)
                    break
        else:
            other_lines.append(line)
    settings_string = "\n".join(settings_lines)
    other_string = "\n".join(other_lines)
    return other_string, settings_string


def _split_yaml_definition(subpic_yaml: _YamlDefinition) -> tuple[str, dict[str, Any]]:
    if isinstance(subpic_yaml, IOBase):
        f = subpic_yaml
        subpic_text = f.readlines()
    else:
        with pathlib.Path(subpic_yaml).open() as f:  # type: ignore[arg-type]
            subpic_text = f.readlines()
    main_file, default_settings_string = split_default_settings_from_yaml(subpic_text)
    if default_settings_string:
        default_settings = yaml.safe_load(default_settings_string)["default_settings"]
    else:
        default_settings = {}
    return main_file, default_settings


def cell_from_yaml_template(
    filename: _YamlDefinition,
    name: str,
    routing_strategy: RoutingStrategies | None = None,
) -> ComponentFactory:
    """Gets a PIC factory function from a yaml definition, which can optionally be a jinja template.

    Args:
        filename: the filepath of the pic yaml template.
        name: the name of the component to create.
        routing_strategy: a dictionary of routing functions.

    Returns:
         a factory function for the component.
    """
    from gdsfactory.pdk import get_routing_strategies

    if routing_strategy is None:
        routing_strategy = get_routing_strategies()
    return yaml_cell(
        yaml_definition=filename, name=name, routing_strategy=routing_strategy
    )


def get_default_settings_dict(
    default_settings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    settings = {}
    for k, v in default_settings.items():
        try:
            v = v["value"]
            if isinstance(v, list):
                v = tuple(v)
            settings[k] = v
        except TypeError as te:  # noqa: PERF203
            raise TypeError(
                f'Default setting "{k}" should be a dictionary with "value" defined.'
            ) from te
        except KeyError as ke:
            raise KeyError(
                f'Required key "value" not supplied for default setting "{k}"'
            ) from ke
    return settings


def yaml_cell(
    yaml_definition: _YamlDefinition, name: str, routing_strategy: RoutingStrategies
) -> ComponentFactory:
    """The "cell" decorator equivalent for yaml files. Generates a proper cell function for yaml-defined circuits.

    Args:
        yaml_definition: the filename to the pic yaml definition.
        name: the name of the pic to create.
        routing_strategy: a dictionary of routing strategies to use for pic generation.

    Returns:
        a dynamically-generated function for the yaml file.
    """
    yaml_body, default_settings_def = _split_yaml_definition(yaml_definition)
    default_settings = get_default_settings_dict(default_settings_def)

    def _yaml_func(**kwargs: Any) -> Component:
        evaluated_text = _evaluate_yaml_template(yaml_body, default_settings, kwargs)
        return _pic_from_templated_yaml(evaluated_text, name, routing_strategy)

    sig = signature(_yaml_func)
    params = []
    docstring_lines = [
        f"{name}: a templated yaml cell. This cell accepts keyword arguments only",
        "",
    ]
    if default_settings:
        docstring_lines.append("Keyword Args:")

    for default_key, default_value in default_settings.items():
        p = Parameter(
            name=default_key, kind=Parameter.KEYWORD_ONLY, default=default_value
        )
        params.append(p)
        description = default_settings_def[default_key].get(
            "description", "No description given"
        )
        docstring_lines.append(f"    {default_key}: {description}")
    new_sig = Signature(parameters=params, return_annotation=sig.return_annotation)
    docstring = "\n".join(docstring_lines)

    _yaml_func.__name__ = name
    _yaml_func.__module__ = "yaml_jinja"
    _yaml_func.__signature__ = new_sig  # type: ignore[attr-defined]
    _yaml_func.__doc__ = docstring
    return cell(_yaml_func)


def _evaluate_yaml_template(
    main_file: str, default_settings: dict[str, Any], settings: dict[str, Any]
) -> str:
    template = jinja2.Template(main_file)
    complete_settings = dict(default_settings)
    complete_settings.update(settings)
    return template.render(**complete_settings)


def _pic_from_templated_yaml(
    evaluated_text: str, name: str, routing_strategy: RoutingStrategies
) -> Component:
    """Creates a component from a  *.pic.yml file.

    This is a lower-level function. See from_yaml_template() for more common usage.

    Args:
        evaluated_text: the text of the yaml file, with all jinja templating evaluated.
        name: the pic name.
        routing_strategy: a dictionary of route factories.

    Returns: the component.
    """
    c = from_yaml(
        evaluated_text,
        routing_strategy=routing_strategy,
    ).dup()
    c.name = name
    return c
