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


dirpath = Path(__file__).parent.joinpath("defaults")
schema_path = dirpath / "default.yaml"
default_yaml = schema_path.read_text()

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
    brand="gdsfactory",
    brand_href="#",
    color="primary",
    dark=True,
    fluid=True,
)

yaml_col = dbc.Col(
    [
        html.H2("input", className="mt-3"),
        dcc.Dropdown(
            options=("netlist", "mask"),
            value="netlist",
            id="dd-input-mode",
            clearable=False,
        ),
        html.H2("YAML", className="mt-3"),
        dbc.Textarea(
            id="yaml_text",
            className="form-control",
            placeholder="Enter the YAML content here...",
            value=default_yaml,
            rows=30,
            spellCheck=False,
            wrap="off",
            persistence=True,
            persistence_type="session",
        ),
        html.Div("", id="yaml_feedback", className="invalid-feedback"),
    ],
    # width=12,
    # xl=6,
)


body = dbc.Container(
    children=[
        dbc.Row(
            [yaml_col],
        )
    ],
    fluid=True,
)
layout = html.Div([navbar, body])
