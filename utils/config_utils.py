import yaml


def load_config(path, default_path=None):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # load configuration from per scene/dataset cfg.
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    inherit_from = cfg_special.get("inherit_from")

    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # merge per dataset cfg. and main cfg.
    update_recursive(cfg, cfg_special)

    return cfg


def apply_overrides(cfg, overrides):
    """
    Applies ``KEY=VALUE`` overrides to a config dict in place.

    Keys use dots to address nested entries, e.g. ``Training.gaussian_th=0.25``.
    Values are parsed as YAML, so scalars keep their type.

    Args:
        cfg (dict): config dict to modify.
        overrides (list): entries of the form ``KEY=VALUE``.

    Returns:
        cfg (dict): the modified config dict.
    """
    for override in overrides:
        key, separator, raw_value = override.partition("=")
        key = key.strip()

        if not separator or not key:
            raise ValueError(
                f"Malformed override {override!r}; expected KEY=VALUE, "
                "e.g. Training.gaussian_th=0.25"
            )

        *parents, leaf = key.split(".")
        node = cfg
        for parent in parents:
            child = node.setdefault(parent, {})
            if not isinstance(child, dict):
                raise ValueError(
                    f"Cannot override {key!r}: {parent!r} is not a mapping"
                )
            node = child
        node[leaf] = yaml.safe_load(raw_value)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively. dict1 get masked by dict2, and we retuen dict1.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v
