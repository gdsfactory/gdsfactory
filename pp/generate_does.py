import collections
from multiprocessing import Process
import time
from omegaconf import OmegaConf

from pp.placer import save_doe
from pp.placer import doe_exists
from pp.placer import build_components
from pp.placer import load_doe_component_names

from pp.config import CONFIG
from pp.components import component_factory
from pp.write_doe import write_doe_metadata
from pp.doe import get_settings_list

from pp.config import logging


def separate_does_from_templates(dicts):
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


def update_dicts_recurse(target_dict, default_dict):
    target_dict = target_dict.copy()
    default_dict = default_dict.copy()
    for k, v in default_dict.items():
        if k not in target_dict:
            target_dict[k] = v
        else:
            if isinstance(target_dict[k], (dict, collections.OrderedDict)):
                target_dict[k] = update_dicts_recurse(target_dict[k], default_dict[k])
    return target_dict


def save_doe_use_template(doe, doe_root_path=None):
    """
    Write a "content.txt" pointing to the DOE used as a template
    """
    doe_name = doe["name"]
    doe_template = doe["doe_template"]
    doe_root_path = doe_root_path or CONFIG["cache_doe_directory"]
    doe_dir = doe_root_path / doe_name
    doe_dir.mkdir(exist_ok=True)
    content_file = doe_dir / "content.txt"

    with open(content_file, "w") as fw:
        fw.write(f"TEMPLATE: {doe_template}")


def write_doe(
    doe,
    component_factory=component_factory,
    doe_root_path=None,
    doe_metadata_path=None,
    regenerate_report_if_doe_exists=False,
    precision=1e-9,
    **kwargs,
):
    doe_name = doe["name"]
    list_settings = doe["list_settings"]

    # Otherwise generate each component using the component factory
    component_type = doe["component"]
    components = build_components(
        component_type, list_settings, component_factory=component_factory
    )

    component_names = [c.name for c in components]
    save_doe(doe_name, components, doe_root_path=doe_root_path, precision=precision)

    write_doe_metadata(
        doe_name=doe["name"],
        cell_names=component_names,
        list_settings=doe["list_settings"],
        doe_settings=kwargs,
        doe_metadata_path=doe_metadata_path,
    )


def load_does(filepath, defaults={"do_permutation": True, "settings": {}}):
    does = {}
    data = OmegaConf.load(filepath)
    data = OmegaConf.to_container(data)
    mask = data.pop("mask")

    for doe_name, doe in data.items():
        for k in defaults:
            if k not in doe:
                doe[k] = defaults[k]

        does[doe_name] = doe
    return does, mask


def generate_does(
    filepath,
    component_factory=component_factory,
    doe_root_path=CONFIG["cache_doe_directory"],
    doe_metadata_path=CONFIG["doe_directory"],
    n_cores=8,
    logger=logging,
    regenerate_report_if_doe_exists=False,
    precision=1e-9,
):
    """Generates a DOEs of components specified in a yaml file
    allows for each DOE to have its own x and y spacing (more flexible than method1)
    similar to write_doe
    """

    doe_root_path.mkdir(parents=True, exist_ok=True)
    doe_metadata_path.mkdir(parents=True, exist_ok=True)

    dicts, mask_settings = load_does(filepath)
    does, templates_by_type = separate_does_from_templates(dicts)

    dict_templates = (
        templates_by_type["template"] if "template" in templates_by_type else {}
    )

    default_use_cached_does = (
        mask_settings["cache"] if "cache" in mask_settings else False
    )

    list_args = []
    for doe_name, doe in does.items():
        doe["name"] = doe_name
        component = doe["component"]

        if component not in component_factory:
            raise ValueError(f"{component} not in {component_factory.keys()}")

        if "template" in doe:
            """
            The keyword template is used to enrich the dictionnary from the template
            """
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

            """
            Only launch a build process if we do not use the cache
            Or if the DOE is not built
            """

            list_settings = doe["list_settings"]

            use_cached_does = (
                default_use_cached_does if "cache" not in doe else doe["cache"]
            )

            _doe_exists = False

            if "doe_template" in doe:
                """
                In that case, the DOE is not built: this DOE points to another existing component
                """
                _doe_exists = True
                logger.info("Using template - {}".format(doe_name))
                save_doe_use_template(doe)

            elif use_cached_does:
                _doe_exists = doe_exists(doe_name, list_settings)
                if _doe_exists:
                    logger.info("Cached - {}".format(doe_name))
                    if regenerate_report_if_doe_exists:
                        component_names = load_doe_component_names(doe_name)

                        write_doe_metadata(
                            doe_name=doe["name"],
                            cell_names=component_names,
                            list_settings=doe["list_settings"],
                            doe_metadata_path=doe_metadata_path,
                        )

            if not _doe_exists:
                start_times[doe_name] = time.time()
                p = Process(
                    target=write_doe,
                    args=(doe, component_factory),
                    kwargs={
                        "doe_root_path": doe_root_path,
                        "doe_metadata_path": doe_metadata_path,
                        "regenerate_report_if_doe_exists": regenerate_report_if_doe_exists,
                        "precision": precision,
                    },
                )
                doe_name_to_process[doe_name] = p
                does_running += [doe_name]
                try:
                    p.start()
                except Exception:
                    print("Issue starting process for {}".format(doe_name))
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
    filepath = CONFIG["samples_path"] / "mask" / "does.yml"
    generate_does(filepath, precision=2e-9)
