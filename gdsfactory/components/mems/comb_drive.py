from __future__ import annotations

__all__ = ["comb_drive"]

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def comb_drive(
    finger_width: float = 0.5,
    finger_length: float = 10.0,
    finger_gap: float = 0.5,
    n_fingers: int = 20,
    finger_overlap: float = 5.0,
    shuttle_width: float = 5.0,
    shuttle_length: float = 30.0,
    spring_width: float = 0.5,
    spring_length: float = 20.0,
    n_spring_folds: int = 4,
    anchor_size: float = 10.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a comb drive actuator with interdigitated fingers and folded springs.

    A central shuttle with comb fingers on both sides, fixed electrodes
    with interleaving fingers, and folded springs connecting the shuttle
    to corner anchor pads.

    Args:
        finger_width: width of each comb finger.
        finger_length: length of each comb finger.
        finger_gap: gap between adjacent moving and fixed fingers.
        n_fingers: number of moving fingers on each side.
        finger_overlap: overlap length between moving and fixed fingers in the actuation direction.
        shuttle_width: width (vertical) of the shuttle mass.
        shuttle_length: length (horizontal) of the shuttle mass.
        spring_width: width of spring beam segments.
        spring_length: length of each spring fold segment.
        n_spring_folds: number of folds in each folded spring.
        anchor_size: size of each square anchor pad.
        layer: layer spec.
    """
    c = Component()

    shw = shuttle_width / 2
    shl = shuttle_length / 2

    # 1. Shuttle rectangle centered at origin
    c.add_polygon(
        [(-shl, -shw), (shl, -shw), (shl, shw), (-shl, shw)],
        layer=layer,
    )

    # Finger pitch (center-to-center of same-side fingers)
    finger_pitch = 2 * (finger_width + finger_gap)

    # Compute total finger array height
    total_finger_height = n_fingers * finger_pitch
    finger_start_y = -total_finger_height / 2

    # 2. Moving fingers extending left and right from shuttle
    for i in range(n_fingers):
        fy = finger_start_y + i * finger_pitch

        # Right-extending moving fingers
        c.add_polygon(
            [
                (shl, fy),
                (shl + finger_length, fy),
                (shl + finger_length, fy + finger_width),
                (shl, fy + finger_width),
            ],
            layer=layer,
        )

        # Left-extending moving fingers
        c.add_polygon(
            [
                (-shl - finger_length, fy),
                (-shl, fy),
                (-shl, fy + finger_width),
                (-shl - finger_length, fy + finger_width),
            ],
            layer=layer,
        )

    # 3. Fixed electrode bars and interleaving fingers
    fixed_bar_width = shuttle_width
    fixed_bar_x_right = shl + 2 * finger_length - finger_overlap
    fixed_bar_x_left = -fixed_bar_x_right

    # Right fixed electrode bar
    c.add_polygon(
        [
            (fixed_bar_x_right, -total_finger_height / 2 - fixed_bar_width),
            (
                fixed_bar_x_right + shuttle_width,
                -total_finger_height / 2 - fixed_bar_width,
            ),
            (
                fixed_bar_x_right + shuttle_width,
                total_finger_height / 2 + fixed_bar_width,
            ),
            (fixed_bar_x_right, total_finger_height / 2 + fixed_bar_width),
        ],
        layer=layer,
    )

    # Left fixed electrode bar
    c.add_polygon(
        [
            (
                fixed_bar_x_left - shuttle_width,
                -total_finger_height / 2 - fixed_bar_width,
            ),
            (fixed_bar_x_left, -total_finger_height / 2 - fixed_bar_width),
            (fixed_bar_x_left, total_finger_height / 2 + fixed_bar_width),
            (
                fixed_bar_x_left - shuttle_width,
                total_finger_height / 2 + fixed_bar_width,
            ),
        ],
        layer=layer,
    )

    # Fixed fingers interleaving with moving fingers
    for i in range(n_fingers):
        fy = finger_start_y + i * finger_pitch + finger_width + finger_gap

        # Right fixed fingers (extending left from right bar)
        c.add_polygon(
            [
                (fixed_bar_x_right - finger_length, fy),
                (fixed_bar_x_right, fy),
                (fixed_bar_x_right, fy + finger_width),
                (fixed_bar_x_right - finger_length, fy + finger_width),
            ],
            layer=layer,
        )

        # Left fixed fingers (extending right from left bar)
        c.add_polygon(
            [
                (fixed_bar_x_left, fy),
                (fixed_bar_x_left + finger_length, fy),
                (fixed_bar_x_left + finger_length, fy + finger_width),
                (fixed_bar_x_left, fy + finger_width),
            ],
            layer=layer,
        )

    # 4. Folded springs connecting shuttle to anchors (top and bottom)
    spring_fold_pitch = spring_width + finger_gap
    for y_sign in [1, -1]:  # top and bottom
        for x_sign in [1, -1]:  # left and right springs
            # Spring attachment point on shuttle
            attach_x = x_sign * shl * 0.5
            attach_y = y_sign * shw

            # Spring extends in y direction away from shuttle
            spring_dir = y_sign

            # Build folded spring segments
            x_cursor = attach_x
            y_cursor = attach_y

            for fold in range(n_spring_folds):
                # Vertical segment
                y_end = y_cursor + spring_dir * spring_length

                c.add_polygon(
                    [
                        (x_cursor - spring_width / 2, min(y_cursor, y_end)),
                        (x_cursor + spring_width / 2, min(y_cursor, y_end)),
                        (x_cursor + spring_width / 2, max(y_cursor, y_end)),
                        (x_cursor - spring_width / 2, max(y_cursor, y_end)),
                    ],
                    layer=layer,
                )

                # Connecting horizontal segment at the end (if not last fold)
                if fold < n_spring_folds - 1:
                    next_x = x_cursor + x_sign * spring_fold_pitch
                    conn_y = y_end

                    c.add_polygon(
                        [
                            (
                                min(x_cursor, next_x) - spring_width / 2,
                                conn_y - spring_width / 2,
                            ),
                            (
                                max(x_cursor, next_x) + spring_width / 2,
                                conn_y - spring_width / 2,
                            ),
                            (
                                max(x_cursor, next_x) + spring_width / 2,
                                conn_y + spring_width / 2,
                            ),
                            (
                                min(x_cursor, next_x) - spring_width / 2,
                                conn_y + spring_width / 2,
                            ),
                        ],
                        layer=layer,
                    )

                    x_cursor = next_x
                    spring_dir = -spring_dir  # Reverse direction for next fold

                y_cursor = y_end

    # 5. Anchor pads at the four corners
    for x_sign in [1, -1]:
        for y_sign in [1, -1]:
            # Compute where the last spring fold ends
            last_x = (
                x_sign * shl * 0.5 + (n_spring_folds - 1) * x_sign * spring_fold_pitch
            )
            if n_spring_folds % 2 == 1:
                last_y = y_sign * (shw + spring_length)
            else:
                last_y = y_sign * shw

            c.add_polygon(
                [
                    (
                        last_x - anchor_size / 2,
                        last_y - anchor_size / 2 * y_sign + anchor_size * y_sign / 2,
                    ),
                    (
                        last_x + anchor_size / 2,
                        last_y - anchor_size / 2 * y_sign + anchor_size * y_sign / 2,
                    ),
                    (
                        last_x + anchor_size / 2,
                        last_y + anchor_size / 2 * y_sign + anchor_size * y_sign / 2,
                    ),
                    (
                        last_x - anchor_size / 2,
                        last_y + anchor_size / 2 * y_sign + anchor_size * y_sign / 2,
                    ),
                ],
                layer=layer,
            )

    return c


if __name__ == "__main__":
    c = comb_drive()
    c.show()
