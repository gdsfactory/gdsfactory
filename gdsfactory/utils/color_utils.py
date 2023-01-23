def ensure_six_digit_hex_color(color: str) -> str:
    if color[0] == "#" and len(color) == 4:
        color = "#" + "".join([2 * str(c) for c in color[1:]])
    return color
