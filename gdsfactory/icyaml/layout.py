"""based on YAMLDash
https://github.com/DrGFreeman/YAMLDash
"""
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import dcc, html

SRC_URL = "https://github.com/gdsfactory/gdsfactory"
YAML_REF_URL = "https://yaml.org/spec/1.2/spec.html"
SCHEMA_REF_URL = "https://json-schema.org/draft/2019-09/json-schema-validation.html"

theme = dbc.themes.UNITED

defaults_path = Path(__file__).parent.joinpath("defaults")

with open(defaults_path.joinpath("default_schema.yaml"), "r") as f:
    default_schema = f.read()

with open(defaults_path.joinpath("default.yaml"), "r") as f:
    default_yaml = f.read()

navbar = dbc.NavbarSimple(
    [
        dbc.NavItem(dbc.NavLink("YAML Reference", href=YAML_REF_URL, target="_blank")),
        dbc.NavItem(
            dbc.NavLink("Schema Reference", href=SCHEMA_REF_URL, target="_blank")
        ),
        dbc.NavItem(
            dbc.NavLink(
                ["Code ", html.I(className="fa fa-lg fa-github")],
                href=SRC_URL,
                target="_blank",
            )
        ),
    ],
    brand="ICYAML",
    brand_href="#",
    color="primary",
    dark=True,
    fluid=True,
)

yaml_col = dbc.Col(
    [
        html.H2("YAML", className="mt-3"),
        dbc.Textarea(
            id="yaml_text",
            className="form-control",
            placeholder="Enter the YAML content here...",
            value=default_yaml,
            rows=20,
            spellCheck=False,
            wrap="off",
            persistence=True,
            persistence_type="session",
        ),
        html.Div("", id="yaml_feedback", className="invalid-feedback"),
    ],
    width=12,
    xl=6,
)

schema_col = dbc.Col(
    [
        html.H2("Schema", id="schema-h2", className="mt-3"),
        dbc.Tooltip(
            "A schema allows validation of the YAML data "
            "against specific requirements.",
            target="schema-h2",
            placement="left",
        ),
        dbc.Textarea(
            id="schema_text",
            placeholder="Enter an optional validation schema here...",
            value=default_schema,
            rows=20,
            spellCheck=False,
            wrap="off",
            persistence=True,
            persistence_type="session",
        ),
        dcc.Store(
            id="schema",
            storage_type="memory",
        ),
        html.Div("", id="schema_feedback", className="invalid-feedback"),
    ],
    width=12,
    xl=6,
)


body = dbc.Container(
    children=[
        dbc.Row(
            [yaml_col, schema_col],
        )
    ],
    fluid=True,
)
layout = html.Div([navbar, body])
