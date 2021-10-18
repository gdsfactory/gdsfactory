import gdsfactory as gf

padding_example0 = """
name: padding_example

instances:
    rectangle:
      component: rectangle
      settings:
        width_mmi: 4.5
        length_mmi: 5
      decorators:
        add_padding:

"""


padding_example1 = """
name: padding_example

decorators:
    add_padding:
        function: add_padding
        settings:
            default: 3

instances:
    rectangle:
      component: rectangle
      settings:
        width_mmi: 4.5
        length_mmi: 5
      decorators:
        add_padding:

"""


if __name__ == "__main__":
    """TODO, fix this"""
    c = gf.read.from_yaml(padding_example1)
    c.show()
