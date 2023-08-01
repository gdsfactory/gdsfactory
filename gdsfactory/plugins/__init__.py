message = """
gdsfactory.plugins have been moved to gplugins

Make sure you have gplugins installed and use gplugins instead of gdsfactory.plugins

You can replace:
    import gdsfactory.plugins -> import gplugins

You can install gplugins with:
    pip install gplugins
"""

raise ImportError(message)
