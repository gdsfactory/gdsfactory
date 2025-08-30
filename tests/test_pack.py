import numpy as np

import gdsfactory as gf


def test_pack() -> None:
    """Test packing function."""
    component_list = [
        gf.components.ellipse(radii=tuple(np.random.rand(2) * n + 2)) for n in range(2)
    ]
    component_list += [
        gf.components.rectangle(size=tuple(np.random.rand(2) * n + 2)) for n in range(2)
    ]

    components_packed_list = gf.pack(
        component_list,  # Must be a list or tuple of Components
        spacing=1.25,  # Minimum distance between adjacent shapes
        aspect_ratio=(2, 1),  # (width, height) ratio of the rectangular bin
        max_size=(None, None),  # Limits the size into which the shapes will be packed
        density=1.05,  # Values closer to 1 pack tighter but require more computation
        sort_by_area=True,  # Pre-sorts the shapes by area
    )
    c = components_packed_list[0]  # Only one bin was created, so we plot that
    assert c


def test_pack_with_settings() -> None:
    """Test packing function with custom settings."""
    component_list = [
        gf.components.rectangle(size=(i, i), port_type=None) for i in range(1, 10)
    ]
    component_list += [
        gf.components.rectangle(size=(i, i), port_type=None) for i in range(1, 10)
    ]

    components_packed_list = gf.pack(
        component_list,  # Must be a list or tuple of Components
        spacing=1.25,  # Minimum distance between adjacent shapes
        aspect_ratio=(2, 1),  # (width, height) ratio of the rectangular bin
        # max_size=(None, None),  # Limits the size into which the shapes will be packed
        max_size=(100, 100),  # Limits the size into which the shapes will be packed
        density=1.05,  # Values closer to 1 pack tighter but require more computation
        sort_by_area=True,  # Pre-sorts the shapes by area
        precision=1e-3,
    )
    assert components_packed_list[0]
