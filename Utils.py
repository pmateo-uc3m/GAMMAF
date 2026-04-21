from types import SimpleNamespace
import yaml

def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    return d

def load_config(args):
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        return dict_to_ns(config_dict)
    else:
        raise ValueError("No configuration file provided. Please specify --config <path_to_yaml>")
    
def load_config_from_path(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return dict_to_ns(config_dict)