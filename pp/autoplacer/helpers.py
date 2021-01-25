import functools
from pathlib import Path
from typing import Union

import klayout.db as pya
from klayout.dbcore import Cell, Layout

CELLS = {}


@functools.lru_cache()
def load_gds(filepath: Union[Path, str]) -> Layout:
    filepath = str(filepath)
    layout = pya.Layout()
    try:
        layout.read(filepath)
    except RuntimeError as e:
        print(f"Error reading {filepath}")
        raise e
    cell = layout.top_cell()
    cell.metadata = {}

    # To make sure the cell does not get destroyed
    global CELLS
    CELLS[cell.name] = layout
    return layout


def import_cell(layout: Layout, cell: Cell) -> Cell:
    """ Imports a cell from another Layout into a given layout"""
    # If the cell is already in the library, skip loading
    if layout.cell(cell.name):
        return layout.cell(cell.name)

    # Create a holder cell and copy in the shapes
    new_cell = layout.create_cell(cell.name)
    new_cell.copy_shapes(cell)

    # Import all the child cells
    for child_index in cell.each_child_cell():
        import_cell(layout, cell.layout().cell(child_index))

    # Import all of the instances, doing the mapping from Layout to Layout
    for instance in cell.each_inst():
        cell_index = layout.cell(instance.cell.name).cell_index()
        # new_instance = pya.CellInstArray(cell_index, instance.trans)
        new_instance = instance.cell_inst.dup()
        new_instance.cell_index = cell_index
        new_cell.insert(new_instance)
    return new_cell
