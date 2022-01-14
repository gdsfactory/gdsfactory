import collections
import multiprocessing
import pathlib
import time
from multiprocessing import Process
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from omegaconf import OmegaConf

from gdsfactory import components
from gdsfactory.config import CONFIG, logger
from gdsfactory.placer import (
    build_components,
    doe_exists,
    load_doe_component_names,
    save_doe,
)
from gdsfactory.sweep.read_sweep import get_settings_list
from gdsfactory.sweep.write_sweep import write_sweep_metadata
from gdsfactory.types import PathType

factory = {
    i: getattr(components, i)
    for i in dir(components)
    if not i.startswith("_") and callable(getattr(components, i))
}

n_cores = multiprocessing.cpu_count()


def separate_does_from_templates(dicts: Dict[str, Any]) -> Any:
    type_to_dict = {}

    does = {}
    for name, d in dicts.items():
        if "type" in d.keys():
            template_type = d.pop("type")
            if template_type not in type_to_dict:
                type_to_dict[template_type] = {}
            type_to_dict[template_type][name] = d
        else:
            does[name] = d

    return does, type_to_dict


def update_dicts_recurse(
    target_dict: Dict[
        str, Union[List[int], str, Dict[str, List[int]], Dict[str, str], bool]
    ],
    default_dict: Dict[str, Union[bool, Dict[str, Union[int, str]], int, str]],
) -> Dict[str, Any]:
    target_dict = target_dict.copy()
    default_dict = default_dict.copy()
    for k, v in default_dict.items():
        if k not in target_dict:
            target_dict[k] = v
        else:
            if isinstance(target_dict[k], (dict, collections.OrderedDict)):
                target_dict[k] = update_dicts_recurse(target_dict[k], default_dict[k])
    return target_dict


def save_doe_use_template(doe, doe_root_path=None) -> None:
    """Write a "content.txt" pointing to the DOE used as a template"""
    doe_name = doe["name"]
    doe_template = doe["doe_template"]
    doe_root_path = doe_root_path or CONFIG["cache_doe_directory"]
    doe_dir = doe_root_path / doe_name
    doe_dir.mkdir(exist_ok=True)
    content_file = doe_dir / "content.txt"

    with open(content_file, "w") as fw:
        fw.write(f"TEMPLATE: {doe_template}")


def write_sweep(
    doe,
    component_factory=factory,
    doe_root_path: Optional[PathType] = None,
    doe_metadata_path: Optional[PathType] = None,
    overwrite: bool = False,
    precision: float = 1e-9,
    **kwargs,
) -> None:
    doe_name = doe["name"]
    list_settings = doe["list_settings"]

    # Otherwise generate each component using the component library
    component_type = doe["component"]
    components = build_components(
        component_type, list_settings, component_factory=component_factory
    )

    component_names = [c.name for c in components]
    save_doe(doe_name, components, doe_root_path=doe_root_path, precision=precision)

    write_sweep_metadata(
        doe_name=doe["name"],
        cell_names=component_names,
        list_settings=doe["list_settings"],
        doe_settings=kwargs,
        doe_metadata_path=doe_metadata_path,
    )


def read_sweep(
    filepath: PathType, defaults: Optional[Dict[str, bool]] = None
) -> Tuple[Any, Any]:
    """Read DOE YAML file and returns a tuple of DOEs"""
    does = {}
    defaults = defaults or {"do_permutation": True, "settings": {}}
    data = OmegaConf.load(filepath)
    data = OmegaConf.to_container(data)
    mask = data.pop("mask")
    data.pop("vars", "")

    for doe_name, doe in data.items():
        for k in defaults:
            if k not in doe:
                doe[k] = defaults[k]

        does[doe_name] = doe
    return does, mask


def write_sweeps(
    filepath: PathType,
    component_factory: Dict[str, Callable] = factory,
    doe_root_path: PathType = CONFIG["cache_doe_directory"],
    doe_metadata_path: PathType = CONFIG["doe_directory"],
    n_cores: int = n_cores,
    overwrite: bool = False,
    precision: float = 1e-9,
    cache: bool = False,
) -> None:
    """Generates a sweep/DOEs of components specified in a yaml file
    allows for each DOE to have its own x and y spacing (more flexible than method1)
    similar to write_doe

    Args:
        filepath: for the does.yml
        component_factory:
        doe_root_path:
        doe_metadata_path:
        n_cores: number of cores
        overwrite:
        precision: for the GDS, defaults to 1nm
        cache: if True uses cache
    """
    doe_root_path = pathlib.Path(doe_root_path)
    doe_metadata_path = pathlib.Path(doe_metadata_path)

    doe_root_path.mkdir(parents=True, exist_ok=True)
    doe_metadata_path.mkdir(parents=True, exist_ok=True)

    dicts, mask_settings = read_sweep(filepath)
    does, templates_by_type = separate_does_from_templates(dicts)

    dict_templates = (
        templates_by_type["template"] if "template" in templates_by_type else {}
    )

    with_cache_default = mask_settings["cache"] if "cache" in mask_settings else cache

    list_args = []
    for doe_name, doe in does.items():
        doe["name"] = doe_name
        component = doe["component"]

        if component not in component_factory:
            raise ValueError(f"{component!r} not in {component_factory.keys()}")

        if "template" in doe:
            # The keyword template is used to enrich the dictionary from the template
            templates = doe["template"]
            if not isinstance(templates, list):
                templates = [templates]
            for template in templates:
                try:
                    doe = update_dicts_recurse(doe, dict_templates[template])
                except Exception:
                    print(template, "does not exist")
                    raise

        do_permutation = doe.pop("do_permutation")
        settings = doe["settings"]
        doe["list_settings"] = get_settings_list(do_permutation, **settings)

        list_args += [doe]

    does_running = []
    start_times = {}
    finish_times = {}
    doe_name_to_process = {}
    while list_args:
        while len(does_running) < n_cores:
            if not list_args:
                break
            doe = list_args.pop()
            doe_name = doe["name"]

            # Only launch a build process if we do not use the cache
            # Or if the DOE is not built

            list_settings = doe["list_settings"]

            with_cache = with_cache_default if "cache" not in doe else doe["cache"]

            _doe_exists = False

            if "doe_template" in doe:
                # this DOE points to another existing component
                _doe_exists = True
                logger.info(f"Using template - {doe_name!r}")
                save_doe_use_template(doe)

            elif with_cache:
                _doe_exists = doe_exists(
                    doe_name=doe_name,
                    list_settings=list_settings,
                    doe_root_path=doe_root_path,
                )
                if _doe_exists:
                    logger.info("Cached - {doe_name!r}")
                    if overwrite:
                        component_names = load_doe_component_names(doe_name)

                        write_sweep_metadata(
                            doe_name=doe["name"],
                            cell_names=component_names,
                            list_settings=doe["list_settings"],
                            doe_metadata_path=doe_metadata_path,
                        )

            if not _doe_exists:
                start_times[doe_name] = time.time()
                p = Process(
                    target=write_sweep,
                    args=(doe, component_factory),
                    kwargs={
                        "doe_root_path": doe_root_path,
                        "doe_metadata_path": doe_metadata_path,
                        "overwrite": overwrite,
                        "precision": precision,
                    },
                )
                doe_name_to_process[doe_name] = p
                does_running += [doe_name]
                try:
                    p.start()
                except Exception:
                    print(f"Issue starting process for {doe_name}")
                    print(type(component_factory))
                    raise

        to_rm = []
        for i, doe_name in enumerate(does_running):
            _p = doe_name_to_process[doe_name]
            if not _p.is_alive():
                to_rm += [i]
                finish_times[doe_name] = time.time()
                dt = finish_times[doe_name] - start_times[doe_name]
                line = "Done - {} ({:.1f}s)".format(doe_name, dt)
                logger.info(line)

        for i in to_rm[::-1]:
            does_running.pop(i)

        time.sleep(0.001)

    while does_running:
        to_rm = []
        for i, _doe_name in enumerate(does_running):
            _p = doe_name_to_process[_doe_name]
            if not _p.is_alive():
                to_rm += [i]
        for i in to_rm[::-1]:
            does_running.pop(i)

        time.sleep(0.05)


if __name__ == "__main__":
    filepath_sample = CONFIG["samples_path"] / "mask" / "does.yml"
    write_sweeps(filepath_sample, precision=2e-9)
