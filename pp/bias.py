from pp.config import CONFIG


def width(width):
    """ return wg_width with bias """
    return width + CONFIG.get("bias", 0)


def gap(gap):
    """ return wg_gap with bias """
    return gap - CONFIG.get("bias", 0)


if __name__ == "__main__":
    # print(width(0.5))
    print(gap(0.5))
