# Deprecation Tracking

This document tracks deprecated features in gdsfactory, their replacements, and removal timeline.

## Deprecation Policy

- Deprecated features emit a `DeprecationWarning` when used.
- Deprecated features are removed in the next major version (currently targeting v10.0.0).
- Use the `@deprecated` decorator from `gdsfactory._deprecation` for new deprecations.

## Active Deprecations

| Feature | Replacement | Since | Removal Target |
|---------|-------------|-------|----------------|
| `generic_tech` module | `gpdk` module | v9.0.0 | v10.0.0 |
| `bend_s(with_euler=...)` | `bend_s(p=0)` for circular, `bend_s(p=1)` for euler | v9.30.0 | v10.0.0 |
| `route_bundle(auto_taper_taper=...)` | `route_bundle(layer_transitions=...)` | v9.35.0 | v10.0.0 |
| `cross_section.strip` | `cross_section.xs_sc` | v9.0.0 | v10.0.0 |
| `cross_section.pin` | See `cross_section.deprecated_pins` | v9.0.0 | v10.0.0 |
| Python 3.10 support | Python 3.11+ | v9.0.0 | v10.0.0 |

## Using the Deprecation System

### For maintainers adding new deprecations:

```python
from gdsfactory._deprecation import deprecated

@deprecated(
    replacement="new_function_name",
    since="9.43.0",
    reason="Superseded by better API",
)
def old_function():
    ...
```

### Checking for overdue deprecations:

```python
from gdsfactory._deprecation import check_deprecations_due

overdue = check_deprecations_due()
if overdue:
    print(f"These features should be removed: {overdue}")
```
