def assert_callable(function):
    if not callable(function):
        raise ValueError(
            f"Error: function = {function} with type {type(function)} is not callable"
        )
