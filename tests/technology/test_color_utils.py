from gdsfactory.technology.color_utils import ensure_six_digit_hex_color


def test_ensure_six_digit_hex_color() -> None:
    assert ensure_six_digit_hex_color(0xFF00FF) == "#ff00ff"
    assert ensure_six_digit_hex_color(0x000000) == "#000000"

    assert ensure_six_digit_hex_color("0xFF00FF") == "#FF00FF"
    assert ensure_six_digit_hex_color("0x000000") == "#000000"

    assert ensure_six_digit_hex_color("#FF00FF") == "#FF00FF"
    assert ensure_six_digit_hex_color("#000000") == "#000000"

    assert ensure_six_digit_hex_color("#0fc") == "#00ffcc"
    assert ensure_six_digit_hex_color("#f0f") == "#ff00ff"


if __name__ == "__main__":
    test_ensure_six_digit_hex_color()
