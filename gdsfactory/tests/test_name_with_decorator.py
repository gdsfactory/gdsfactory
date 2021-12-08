import gdsfactory as gf


@gf.cell
def straight_with_pins(**kwargs):
    c = gf.Component()
    ref = c << gf.c.straight()
    c.add_ports(ref.ports)
    return c


def test_name_with_decorator():
    c = gf.Component("test_name_with_decorator")
    c1 = c << straight_with_pins(decorator=gf.add_padding_container)
    c2 = c << straight_with_pins()

    c1.movey(-10)
    c2.movey(100)

    cells = c.get_dependencies()
    cell_names = [cell.name for cell in list(cells)]
    cell_names_unique = set(cell_names)

    if len(cell_names) != len(set(cell_names)):
        for cell_name in cell_names_unique:
            cell_names.remove(cell_name)

        cell_names_duplicated = "\n".join(set(cell_names))
        raise ValueError(f"Duplicated cell names in {c.name}:\n{cell_names_duplicated}")

    referenced_cells = list(c.get_dependencies(recursive=True))
    all_cells = [c] + referenced_cells

    no_name_cells = [cell.name for cell in all_cells if cell.name.startswith("Unnamed")]
    assert (
        len(no_name_cells) == 0
    ), f"Component {c.name} contains {len(no_name_cells)} Unnamed cells"


if __name__ == "__main__":
    c = gf.Component("test_name_with_decorator")
    c1 = c << straight_with_pins(decorator=gf.add_padding_container)
    c2 = c << straight_with_pins()
    c1.movey(-10)
    c2.movey(100)

    cells = c.get_dependencies()
    cell_names = [cell.name for cell in list(cells)]
    cell_names_unique = set(cell_names)

    if len(cell_names) != len(set(cell_names)):
        for cell_name in cell_names_unique:
            cell_names.remove(cell_name)

        cell_names_duplicated = "\n".join(set(cell_names))
        raise ValueError(f"Duplicated cell names in {c.name}:\n{cell_names_duplicated}")

    referenced_cells = list(c.get_dependencies(recursive=True))
    all_cells = [c] + referenced_cells

    no_name_cells = [cell.name for cell in all_cells if cell.name.startswith("Unnamed")]
    assert (
        len(no_name_cells) == 0
    ), f"Component {c.name} contains {len(no_name_cells)} Unnamed cells"
