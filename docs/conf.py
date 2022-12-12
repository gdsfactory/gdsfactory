project = "gdsfactory"
release = "6.8.2"
copyright = "2020, MIT License"

html_theme = "sphinx_book_theme"
html_logo = "logo.png"


extensions = [
    "matplotlib.sphinxext.plot_directive",
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_click",
    "sphinx_markdown_tables",
    "sphinx_copybutton",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
]

autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "build",
    "extra",
]

napoleon_use_param = True
nbsphinx_timeout = 300

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

language = "en"
myst_html_meta = {
    "description lang=en": "metadata description",
    "keywords": "Sphinx, MyST",
    "property=og:locale": "en_US",
}


html_theme_options = {
    "logo_only": True,
    "path_to_docs": "docs",
    "repository_url": "https://github.com/gdsfactory/gdsfactory",
    "repository_branch": "main",
    "launch_buttons": {
        "notebook_interface": "jupyterlab",
        "binderhub_url": "https://mybinder.org/v2/gh/gdsfactory/gdsfactory/HEAD",
        "colab_url": "https://colab.research.google.com",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}

autodoc_pydantic_model_signature_prefix = "class"
autodoc_pydantic_field_signature_prefix = "attribute"
autodoc_pydantic_model_show_config_member = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_typehints = "description"
autodoc_typehints_format = "short"

autodoc_type_aliases = {
    "ComponentSpec": "ComponentSpec",
    "LayerSpec": "LayerSpec",
    "CrossSectionSpec": "CrossSectionSpec",
}

autodoc_default_options = {
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "inherited-members": True,
    "show-inheritance": True,
}
