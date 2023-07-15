import pytest
from gdsfactory.geometry.write_connectivity import write_connectivity_checks

nm = 1e-3


def test_default_values() -> None:
    script = write_connectivity_checks(pin_widths=[0.5, 0.9, 0.45], pin_layer=(1, 10))
    assert isinstance(script, str)
    assert "pin = input(1, 10)" in script
    width = 0.5
    print(script)
    assert f"pin2 = pin.rectangles.without_area({width} * 0.002)" in script
    assert 'pin2.output("port alignment error")' in script
    assert 'pin2.non_rectangles.output("port width check")' in script


# Tests that the function works with different pin widths
def test_different_pin_widths() -> None:
    width = 0.3
    script = write_connectivity_checks(pin_widths=[width], pin_layer=(1, 10))
    assert isinstance(script, str)
    assert "pin = input(1, 10)" in script
    assert f"pin.rectangles.without_area({width} * 0.002)" in script


# Tests that the function works with different pin layers
def test_different_pin_layers() -> None:
    script = write_connectivity_checks(pin_widths=[0.5, 0.9, 0.45], pin_layer=(1, 20))
    assert isinstance(script, str)
    assert "pin = input(1, 20)" in script


# Tests that the function works with different pin lengths
def test_different_pin_lengths() -> None:
    script = write_connectivity_checks(
        pin_widths=[0.5, 0.9, 0.45], pin_layer=(1, 10), pin_length=2 * nm
    )
    assert isinstance(script, str)
    assert "pin = input(1, 10)" in script
    assert "pin2 = pin.rectangles.without_area(0.5 * 0.004)" in script


# Tests that the function raises an error with an invalid pin layer
def test_invalid_pin_layer() -> None:
    with pytest.raises(ValueError):
        write_connectivity_checks(
            pin_widths=[0.5, 0.9, 0.45], pin_layer="wrong_layer_name"
        )


# Tests that the function raises an error with an invalid device layer
def test_invalid_device_layer() -> None:
    with pytest.raises(ValueError):
        write_connectivity_checks(
            pin_widths=[0.5, 0.9, 0.45],
            pin_layer=(1, 10),
            device_layer="wrong_layer_name",
        )


if __name__ == "__main__":
    # s = write_connectivity_checks(pin_widths=[0.5, 0.9, 0.45], pin_layer='wrong')
    # print(s)
    # script = write_connectivity_checks(
    #     pin_widths=[0.5, 0.9, 0.45], pin_layer=(1, 10), pin_length=2 * nm
    # )
    # print(script)
    # test_default_values()
    # test_different_pin_widths()
    # test_different_pin_layers()
    test_different_pin_lengths()
