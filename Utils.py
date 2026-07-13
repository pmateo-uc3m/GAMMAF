import yaml


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


_TRAINING_DEFAULTS = {
    "early_stop": 10,
    "lr_patience_max": 5,
}


def _to_attrdict(d):
    if isinstance(d, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in d.items()})
    return d


def _apply_training_defaults(cfg):
    for k, v in _TRAINING_DEFAULTS.items():
        if k not in cfg:
            cfg[k] = v
    return cfg


def load_config(args):
    if args.config:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
        return _to_attrdict(config_dict)
    raise ValueError("No configuration file provided. Please specify --config <path_to_yaml>")


def load_config_from_path(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = _to_attrdict(config_dict)
    cfg = _apply_training_defaults(cfg)
    if cfg.get("lr") is None and cfg.get("learning_rate") is not None:
        cfg.lr = cfg.learning_rate
    if cfg.get("epochs") is None and cfg.get("num_epochs") is not None:
        cfg.epochs = cfg.num_epochs
    return cfg
