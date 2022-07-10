import gdsfactory


@gdsfactory.cell
def h_tree(
    level, length, straight_factory, splitter_factory, bend_factory, leaf_factory
):
    """
    Generates an H-Tree

    Args:
        level: number of levels
        length: length of the cell at the lowest level
        straight_factory: Factory for the straight element
        splitter_factory: Factory for the splitter
        bend_factory: Factory for the bend
        leaf_factory: Factory for the leafs of the Tree

    Returns:
        H-Tree component
    """
    tree = gdsfactory.Component()
    splitter = splitter_factory()
    bend = bend_factory()

    splitter_length = (
        splitter.ports["o2"].midpoint[0] - splitter.ports["o1"].midpoint[0]
    )
    straight = tree.add_ref(
        straight_factory(
            2 ** (level // 2) * length - splitter_length - 2 * bend.info["radius"]
        )
    )
    tree.add_ports([straight.ports["o1"]])

    splitter_ref = tree.add_ref(splitter).connect("o1", straight.ports["o2"])
    bend_left = tree.add_ref(bend).connect("o1", splitter_ref.ports["o2"])
    bend_right = tree.add_ref(bend).connect("o2", splitter_ref.ports["o3"])

    if level > 1:
        leaf = h_tree(
            level - 1,
            length,
            straight_factory,
            splitter_factory,
            bend_factory,
            leaf_factory,
        )
    else:
        leaf = leaf_factory()

    tree.add_ref(leaf).connect("o1", bend_left.ports["o2"])
    tree.add_ref(leaf).connect("o1", bend_right.ports["o1"])

    return tree


if __name__ == "__main__":
    component = h_tree(
        10,
        50,
        gdsfactory.components.straight,
        gdsfactory.components.mmi1x2,
        gdsfactory.components.bend_circular,
        gdsfactory.components.grating_coupler_te,
    )

    main_component = gdsfactory.Component("main")
    main_component.add_ref(component)
    main_component.write_gds("h_tree.gds")
