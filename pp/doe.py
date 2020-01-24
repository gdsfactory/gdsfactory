""" A Design Of Experiment (DOE) changes one or several component parameters to create a model
"""

import itertools as it
from pp.config import CONFIG, load_config


sample = """
does:
    cutback_taper_te_400:
      component: cutback_taper_te
      settings:
        wg_width: [0.4]
        length: [1.5]
        n_devices_target: [30, 60, 120]

"""


def load_does(config=CONFIG):
    """ returns a dictionary with the information loaded from does.yml

    Args:
        filepath: yaml file describes does

    Returns:
        a dictionnary of DOEs with:
        {
            doe_name1: [(component_factory_name, parameters), ...]
            doe_name2: [(component_factory_name, parameters), ...]
            ...
        }

    .. code:: python

        mmi1x2_gap:
            component: mmi1x2
            gap: [0.5, 0.6]
            length: 10

        mmi1x2:
            component: mmi1x2
            length: [11, 12]
            gap: [0.2, 0.3]
            do_permutation: False

    """
    does = {}
    if config.get("does") is None:
        raise ValueError(f"no does defined in {CONFIG}")
    input_does = config["does"]

    for doe_name, doe in input_does.items():
        if "do_permutation" in doe:
            do_permutation = doe.pop("do_permutation")
        else:
            do_permutation = True
        assert doe.get("settings"), "need to define settings for doe {}".format(
            doe_name
        )

        doe_settings = doe.pop("settings")
        try:
            doe["settings"] = get_settings_list(do_permutation, **doe_settings)
        except Exception:
            print(doe_name, "DOE needs to be a dictionary")
            print("got:", doe)
            print("sample:", sample)

            raise
        does[doe_name] = doe

    return does


def get_settings_list(do_permutations=True, **kwargs):
    """
    Returns a list of settings

    Args:
        do_permutations: if False, will only zip the values passed for each parameter
        and will not use any combination with default arguments
        **kwargs: Keyword arguments with a list or tuple of desired values to sweep

    Usage:

        import pp

        pp.doe.get_settings_list(length=[30, 40])  # adds different lengths
        pp.doe.get_settings_list(length=[30, 40], width=[4, 8])  # if do_permutations=False, zips arguments (L30W4, L40W8)
        pp.doe.get_settings_list(length=[30, 40], width=[4, 8])  # if do_permutations=True, does all combinations (L30W4, L30W8, L40W4, L40W8)


    add variations of self.baseclass in self.components
    get arguments from default_args and then update them from kwargs
    updates default_args with kwargs
    self.settings lists all the variations
    """

    # Deal with empty parameter case
    if kwargs == {}:
        return {}

    # Accept both values or lists
    for key, value in list(kwargs.items()):
        if not isinstance(value, list):
            kwargs[key] = [value]

    if do_permutations:
        keys, list_values = list(zip(*[x for x in list(kwargs.items())]))
        settings = [dict(list(zip(keys, perms))) for perms in it.product(*list_values)]
    else:
        keys, list_values = list(zip(*[x for x in list(kwargs.items())]))
        settings = [dict(list(zip(keys, values))) for values in zip(*list_values)]

    return settings


def test_load_doe():
    config = load_config(CONFIG["samples_path"] / "mask" / "config.yml")
    does = load_does(config)
    assert does
    return does


if __name__ == "__main__":
    # test_load_doe()
    from pprint import pprint

    config = load_config(CONFIG["samples_path"] / "mask" / "config.yml")
    pprint(load_does(config))
