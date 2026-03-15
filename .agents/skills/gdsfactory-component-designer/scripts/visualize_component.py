#!/usr/bin/env python3
"""Render a gdsfactory component to a PNG image.

Usage
-----
    python visualize_component.py <component_expression> <output_path> [--width W] [--height H]

Depending on the user's environment you may need to invoke this script with a
different Python command, for example::

    uv run python visualize_component.py ...
    python3 visualize_component.py ...

Choose whichever command gives access to the ``gdsfactory`` package in the
current project.

Examples
--------
    python visualize_component.py "gf.components.mmi1x2()" /tmp/mmi1x2.png
    python visualize_component.py "gf.components.bend_euler(radius=15)" /tmp/bend.png --width 1024 --height 768

The script activates the generic PDK, evaluates the expression to obtain a
Component, renders it with KLayout, and writes a PNG file to *output_path*.
A short textual summary (name, ports, bounding-box) is printed to stdout so
the calling agent can include it in context without opening the image.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a gdsfactory component to PNG."
    )
    parser.add_argument(
        "expression",
        help=(
            "Python expression that returns a gf.Component. "
            "Example: gf.components.mmi1x2(width_mmi=5)"
        ),
    )
    parser.add_argument("output", help="Destination PNG file path.")
    parser.add_argument(
        "--width", type=int, default=800, help="Image width in pixels (default 800)."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="Image height in pixels (default 600).",
    )
    args = parser.parse_args()

    # Force headless matplotlib backend before any other import touches it.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import gdsfactory as gf

    # Activate the built-in generic PDK so components can be resolved.
    gf.gpdk.PDK.activate()
    gf.clear_cache()

    # Evaluate the user-provided expression in a restricted namespace that only
    # exposes the gdsfactory module.  __builtins__ is explicitly blanked so that
    # arbitrary built-in functions (open, __import__, exec, …) are unavailable.
    restricted_ns: dict[str, object] = {"__builtins__": {}, "gf": gf}
    try:
        component = eval(args.expression, restricted_ns)
    except Exception as exc:
        print(f"ERROR: Failed to evaluate expression: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(component, gf.Component):
        print(
            f"ERROR: Expression did not return a Component (got {type(component).__name__}).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Render to PNG via Component.plot().
    fig = component.plot(
        return_fig=True,
        pixel_buffer_options={"width": args.width, "height": args.height},
    )
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print a concise summary for the agent.
    port_names = [p.name for p in component.ports]
    print(f"Component : {component.name}")
    print(f"Ports     : {port_names}")
    print(f"Bbox size : {component.dxsize:.3f} x {component.dysize:.3f} µm")
    print(f"Image     : {args.output}")


if __name__ == "__main__":
    main()
