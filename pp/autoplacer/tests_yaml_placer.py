from pp.autoplacer.yaml_placer import update_dicts_recurse


def test1() -> None:
    target_dict = {
        "placer": {"x0": 1, "y0": 1},
        "nested": {"nested1": {"nested2": {"toto": "toto"}}},
    }

    default_dict = {
        "placer": {"x0": 2, "y0": 2, "x": 0, "y": 200},
        "name": {"suffix": "TEG1"},
        "nested": {"nested1": {"nested2": {"foo": "bar"}}},
    }

    new_dict = update_dicts_recurse(target_dict.copy(), default_dict.copy())
    print(new_dict)


if __name__ == "__main__":
    test1()
