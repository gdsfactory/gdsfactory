"""based on YAMLDash
https://github.com/DrGFreeman/YAMLDash
"""

import webbrowser
from multiprocessing import cpu_count
from pathlib import Path

import dash
import jsonschema
import yaml
from dash.dependencies import Input, Output

from gdsfactory.icyaml.layout import layout, theme
from gdsfactory.read.from_yaml import from_yaml

ascii_title = "YAML IC"

app = dash.Dash(
    __name__,
    external_stylesheets=[
        theme,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
    ],
)
defaults_path = Path(__file__).parent.joinpath("defaults")

with open(defaults_path.joinpath("schema.yaml"), "r") as f:
    schema_text = f.read()

schema_dict = yaml.safe_load(schema_text)

wsgi_app = app.server
app.title = "ICYAML- Interactive IC based YAML Validator"
app.layout = layout


def run_debug():
    app.run_server(debug=True)
    # app.run_server(debug=True)


def run():

    print(ascii_title)
    webbrowser.open("127.0.0.1:8080", new=2)

    try:
        import waitress

        print("Listening on 127.0.0.1:8080.")
        print("Press CTRL-C to stop.")
        waitress.serve(wsgi_app, listen="127.0.0.1:8080", threads=cpu_count())

    except ModuleNotFoundError:
        print("Waitress server not found (use 'pip install waitress' to install it)")
        print("Defaulting to Flask development server.\n")

        app.run_server(port=8080)


@app.callback(
    [Output("yaml_text", "className"), Output("yaml_feedback", "children")],
    [Input("yaml_text", "value")],
)
def validate_yaml(yaml_text, schema=schema_dict):
    class_name = "form-control"

    try:
        if yaml_text != "" and yaml_text is not None:
            yaml_dict = yaml.safe_load(yaml_text)
        else:
            return class_name, ""
    except Exception as e:
        return (class_name + " is-invalid", f"YAML ParsingError: {e}")

    if yaml_dict is not None:
        try:
            jsonschema.validate(yaml_dict, schema)
            c = from_yaml(yaml_text)
            c.show()
            return class_name + " is-valid", ""
        except ValueError as e:
            return (class_name + " is-invalid", f"ValueError {e}")
        except jsonschema.exceptions.ValidationError as e:
            return (class_name + " is-invalid", f"Schema ValidationError: {e}")
    else:
        return class_name, ""
