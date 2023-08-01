message = """
gdsfactory.simulation have been moved to gplugins

Make sure you have gplugins installed and use gplugins instead of gdsfactory.simulation

You can replace:
    import gdsfactory.simulation -> import gplugins

You can install gplugins with:
    pip install gplugins
"""


raise ValueError(message)
