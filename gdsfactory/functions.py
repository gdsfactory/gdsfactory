from functools import lru_cache

cache = lru_cache


def copy_settings(component_old, component_new) -> None:
    """Propagate_settings from one old component to new component"""
    component_new.info["parent_name"] = component_old.get_parent_name()
    component_new.info["parent"] = component_old.get_settings()
    # component_new.info["parent_settings"] = component_old.settings
    # component_new.info["parent_info"] = component_old.info
