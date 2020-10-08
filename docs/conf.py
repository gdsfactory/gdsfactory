from recommonmark.transform import AutoStructify

project = "gdsfactory"
version = "1.4.2"
copyright = "2019, PsiQ"
author = "PsiQ"

master_doc = "index"
html_theme = "sphinx_rtd_theme"

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

htmlhelp_basename = project

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_markdown_tables",
    "recommonmark",
    "sphinx_autodoc_typehints",
]

napoleon_use_param = True


def setup(app):
    app.add_config_value(
        "recommonmark_config",
        {"auto_toc_tree_section": "Contents", "enable_eval_rst": True},
        True,
    )
    app.add_transform(AutoStructify)
