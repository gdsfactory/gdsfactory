import phidl.geometry as pg
from pp.import_phidl_component import import_phidl_component


def boolean(
    A, B, operation, precision=1e-4, num_divisions=[1, 1], max_points=4000, layer=0
):
    """
    Performs boolean operations between 2 Device/DeviceReference objects,
    or lists of Devices/DeviceReferences.

    ``operation`` should be one of {'not', 'and', 'or', 'xor', 'A-B', 'B-A', 'A+B'}.
    Note that 'A+B' is equivalent to 'or', 'A-B' is equivalent to 'not', and
    'B-A' is equivalent to 'not' with the operands switched
    """
    c = pg.boolean(
        A=A,
        B=B,
        operation=operation,
        precision=precision,
        num_divisions=num_divisions,
        max_points=max_points,
        layer=layer,
    )
    return import_phidl_component(component=c)


def _demo():
    import pp

    e1 = pp.c.ellipse()
    e2 = pp.c.ellipse(radii=(10, 6)).movex(2)
    e3 = pp.c.ellipse(radii=(10, 4)).movex(5)
    pp.qp([e1, e2, e3])
    c = boolean(A=[e1, e3], B=e2, operation="A-B")
    pp.show(c)


if __name__ == "__main__":

    _demo()
