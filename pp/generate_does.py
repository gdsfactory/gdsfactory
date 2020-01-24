import os
from pp.placer import save_doe
from pp.placer import doe_exists
from pp.placer import _gen_components
from pp.placer import _gen_components_from_generator
from pp.placer import load_doe_component_names

from pp.config import CONFIG
from pp.components import component_type2factory
from pp.write_doe import write_doe_report
from multiprocessing import Process
import time
import hiyapyco
from pp.doe import get_settings_list

from pp.logger import LOGGER
import sys
import collections
from pprint import pprint


def _print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def _pprint(*args, **kwargs):
    pprint(*args, **kwargs)
    sys.stdout.flush()


def _test(doe_params):
    print(doe_params[0])


def default_component_filter(x):
    return x


def _generate_doe_report(doe, component_names, doe_json_root_path=None):
    if doe_json_root_path is None:
        doe_json_root_path = os.path.join(CONFIG["build_directory"], "devices")

    doe_name = doe["name"]
    print("Generating report {}".format(doe_name))
    description = doe["description"] if "description" in doe else ""
    test = doe["test"] if "test" in doe else ""
    analysis = doe["analysis"] if "analysis" in doe else ""

    doe_settings = {"description": description, "test": test, "analysis": analysis}
    list_settings = doe["list_settings"]

    json_doe_path = os.path.join(doe_json_root_path, doe_name + ".json")
    # Write the json and md metadata / report
    write_doe_report(
        doe_name=doe_name,
        cell_names=component_names,
        list_settings=list_settings,
        doe_settings=doe_settings,
        json_doe_path=json_doe_path,
    )


def separate_does_from_templates(dicts):
    type_to_dict = {}

    does = {}
    for name, d in dicts.items():
        if "type" in d.keys():
            template_type = d.pop("type")
            if not template_type in type_to_dict:
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
            vtype = type(target_dict[k])
            if vtype == dict or vtype == collections.OrderedDict:
                target_dict[k] = update_dicts_recurse(target_dict[k], default_dict[k])
    return target_dict


def save_doe_use_template(doe, doe_root_path=None):
    """
    Write a "content.txt" pointing to the DOE used as a template
    """
    doe_name = doe["name"]
    doe_template = doe["doe_template"]
    if doe_root_path is None:
        doe_root_path = CONFIG["cache_doe_directory"]
    doe_dir = os.path.join(doe_root_path, doe_name)
    if not os.path.exists(doe_dir):
        os.makedirs(doe_dir)
    content_file = os.path.join(doe_dir, "content.txt")
    with open(content_file, "w") as fw:
        fw.write("TEMPLATE: {}".format(doe_template))


def _generate_doe(
    doe,
    component_type2factory=component_type2factory,
    component_filter=default_component_filter,
    doe_json_root_path=None,
    doe_root_path=None,
    regenerate_report_if_doe_exists=False,
    precision=1e-9,
    logger=LOGGER,
):
    doe_name = doe["name"]
    list_settings = doe["list_settings"]

    line = "Building - {} ...".format(doe_name)
    logger.info(line)

    # Otherwise generate each component using the component factory
    component_type = doe["component"]
    if "generator" in doe:
        components = _gen_components_from_generator(
            doe["generator"],
            component_type,
            doe,
            component_type2factory=component_type2factory,
        )
    else:
        components = _gen_components(
            component_type, list_settings, component_type2factory=component_type2factory
        )

    components = [component_filter(c) for c in components]
    component_names = [c.name for c in components]
    save_doe(doe_name, components, doe_root_path=doe_root_path, precision=precision)

    _generate_doe_report(doe, component_names, doe_json_root_path)


def load_does(filepath, defaults={"do_permutation": True, "settings": {}}):
    does = {}
    data = hiyapyco.load(filepath)
    mask = data.pop("mask")

    for doe_name, doe in data.items():
        for k in defaults:
            if k not in doe:
                doe[k] = defaults[k]

        does[doe_name] = doe
    return does, mask


def generate_does(
    filepath,
    component_filter=default_component_filter,
    component_type2factory=component_type2factory,
    doe_root_path=None,
    doe_json_root_path=None,
    n_cores=4,
    logger=LOGGER,
    regenerate_report_if_doe_exists=False,
    precision=1e-9,
):
    """ Returns a Component composed of DOEs/components given in a yaml file
    allows for each DOE to have its own x and y spacing (more flexible than method1)
    """

    if doe_root_path is None:
        doe_root_path = CONFIG["cache_doe_directory"]

    if doe_json_root_path is None:
        doe_json_root_path = os.path.join(CONFIG["build_directory"], "devices")

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

        if "template" in doe:
            """
            The keyword template is used to enrich the dictionnary from the template
            """
            templates = doe["template"]
            if type(templates) != list:
                templates = [templates]
            for template in templates:
                try:
                    doe = update_dicts_recurse(doe, dict_templates[template])
                except:
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
                        _generate_doe_report(doe, component_names, doe_json_root_path)

            if not _doe_exists:
                start_times[doe_name] = time.time()
                p = Process(
                    target=_generate_doe,
                    args=(doe, component_type2factory),
                    kwargs={
                        "component_filter": component_filter,
                        "doe_root_path": doe_root_path,
                        "doe_json_root_path": doe_json_root_path,
                        "regenerate_report_if_doe_exists": regenerate_report_if_doe_exists,
                        "precision": precision,
                        "logger": logger,
                    },
                )
                doe_name_to_process[doe_name] = p
                does_running += [doe_name]
                try:
                    p.start()
                except:
                    print("Issue starting process for {}".format(doe_name))
                    print(type(component_type2factory))
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
