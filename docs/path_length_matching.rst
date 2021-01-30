Path length matching
========================

For using instead of `get_bundle` where path length matching matters.

.. autofunction:: pp.routing.get_bundle.get_bundle_path_length_match



Tips:

- If the path length matching is done on the wrong segments, change `modify_segment_i` arguments.
- Adjust `nb_loops` to avoid too short or too long segments
- Adjust `separation` and `end_straight_offset` to avoid path length compensation collisions


.. autofunction:: pp.routing.path_length_matching.path_length_matched_points
