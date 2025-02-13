from unittest.mock import Mock

import pytest

import gdsfactory as gf


def test_array_errors() -> None:
    with pytest.raises(ValueError, match="rows = 2 > 1 require row_pitch=0 > 0"):
        gf.components.array(rows=2, row_pitch=0)

    with pytest.raises(ValueError, match="columns = 2 > 1 require 0 > 0"):
        gf.components.array(columns=2, column_pitch=0)


def test_array_size() -> None:
    column_pitch = 150.0
    row_pitch = 150.0
    size = (450.0, 300.0)

    c1 = gf.components.array(size=size, column_pitch=column_pitch, row_pitch=row_pitch)

    c2 = gf.components.array(
        columns=int(size[0] / column_pitch),
        rows=int(size[1] / row_pitch),
        column_pitch=column_pitch,
        row_pitch=row_pitch,
    )

    assert len(c1.insts) == len(c2.insts)


def test_array_post_process() -> None:
    mock_post_process = Mock()

    def my_post_process(component: gf.Component) -> None:
        mock_post_process(component)

    gf.components.array(post_process=[my_post_process])
    mock_post_process.assert_called_once()


def test_array_auto_rename_ports() -> None:
    c1 = gf.components.array()
    c2 = gf.components.array(auto_rename_ports=True)
    assert len(c1.ports) == len(c2.ports)
    assert {p.name for p in c1.ports} != {p.name for p in c2.ports}


if __name__ == "__main__":
    pytest.main([__file__])
