Path length matching
========================

Low level function
------------------------
For path length matching a list of lists of waypoints.

.. autofunction:: pp.routing.path_length_matching.path_length_matched_points

High level function
------------------------
For using instead of `connect_bundle` where path length matching matters.

Tips:

- If the path length matching is done on the wrong segments, change `modify_segment_i` arguments.
- Adjust `nb_loops` to avoid too short or too long segments
- Adjust `separation` and `end_straight_offset` to avoid path length compensation collisions

.. autofunction:: pp.routing.connect_bundle.connect_bundle_path_length_match
