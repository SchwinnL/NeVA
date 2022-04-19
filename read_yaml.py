import yaml
from global_config import enums, all_enums, train_keys

def open_yaml(file_name):
    with open("yaml/" + file_name + ".yaml") as f:
        arg_dict = yaml.load(f, Loader=yaml.FullLoader)

    for key in enums:
        if key in arg_dict:
            enum = convert_to_enum(arg_dict[key])
            arg_dict[key] = enum

    arguments = []
    for key in arg_dict:
        en = arg_dict[key]
        if type(en) is not list:
            en = [en]
        arguments.append(en)

    return arguments, list(arg_dict.keys())

def convert_to_enum(data):
    def convert(data):
        if isinstance(data, list):
            arr = []
            for d in data:
                val = convert(d)
                arr.append(val)
        elif isinstance(data, dict):
            arr = {}
            for key in data:
                val = convert(data[key])
                arr[key] = val
        else:
            for enum in all_enums:
                if data in enum.__members__:
                    return enum[data]
            return data
        return arr
    enum = convert(data)
    return enum

def convert_to_str(data, enum):
    if isinstance(data, list):
        ret = []
        for d in data:
            if isinstance(d, list):
                inner = []
                for d2 in d:
                    inner.append(str(d2))
                ret.append(inner)
            else:
                ret.append(str(d))
        return ret
    else:
        return str(data)

def write_yaml(file_name, config):
    # convert Enums back to their string representation
    for key, enum in enums.items():
        config[key] = convert_to_str(config[key], enum)

    with open(file_name, 'w') as file:
        # dump the yaml file, default_flow_style=None will try to put lists in one line
        yaml.dump(config, file, sort_keys=False, default_flow_style=None)
