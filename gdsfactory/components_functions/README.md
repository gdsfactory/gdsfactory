# Component Functions

This module contains pure functions that create components without any decorators.
This pattern helps avoid double-decorated functions and makes the code more modular and testable.

## Pattern

### Before (in components/):
```python
@gf.cell_with_module_name
def my_component(param1: float = 1.0, param2: str = "default") -> Component:
    """Component with decorators."""
    # implementation
    c = Component()
    # ... component logic ...
    return c
```

### After:

#### In component_functions/:
```python
def my_component_function(param1: float = 1.0, param2: str = "default") -> Component:
    """Pure function without decorators."""
    # implementation
    c = Component()
    # ... component logic ...
    return c
```

#### In components/:
```python
from gdsfactory.component_functions.module import my_component_function

@gf.cell_with_module_name
def my_component(param1: float = 1.0, param2: str = "default") -> Component:
    """Component with decorator that calls the pure function."""
    return my_component_function(param1=param1, param2=param2)
```

## Benefits

1. **Single Responsibility**: Functions do one thing - create components
2. **Testability**: Pure functions are easier to test without decorator side effects
3. **Reusability**: Functions can be used with different decorators or no decorators
4. **Clarity**: Clear separation between component logic and caching/naming logic

## Structure

```
component_functions/
├── __init__.py          # Main exports
├── README.md            # This file
├── waveguides.py        # Waveguide functions
├── bends.py             # Bend functions
├── couplers.py          # Coupler functions
└── ...                  # Other component categories
```

## Migration Guide

To migrate a double-decorated function:

1. Copy the function body to the appropriate module in `component_functions/`
2. Remove all decorators
3. Rename the function with a `_function` suffix
4. Update the original function to call the new pure function
5. Keep only the necessary decorator(s) on the original function
6. Update imports in `__init__.py`

This keeps backward compatibility while cleaning up the code structure.
