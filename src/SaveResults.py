import os

def contains_result(conf, key):
    path = conf.result_save_path(conf.model.name + "_metrics")
    if os.path.isfile(path):
        old_results = open(path, "r").read()
        old_results_dict = generate_result_dict(old_results)
        if key in old_results_dict:
            if isinstance(old_results_dict[key], str) and "/" in old_results_dict[key]:
                return False
            else:
                return True
    return False

def save_result_dict(conf, result_dict, name=""):
    path = conf.result_save_path(name)
    new_results = generate_result_txt(result_dict)
    print(path.split("/")[-1])
    print("New Results:\n" + new_results)
    dict = {}
    if os.path.isfile(path):
        old_results = open(path, "r").read()
        old_result_dict = generate_result_dict(old_results)
        dict.update(old_result_dict)
    dict.update(result_dict)
    results = generate_result_txt(dict)
    open(path, "w").write(results)

def generate_result_dict(results):
    results_dict = {}
    split = results.strip().split("\n")
    for line in split:
        if ":" in line:
            key = line.split(":")[0]
            value = line.split(":")[1].strip()
            if is_float(value):
                results_dict[key] = float(value)
            else:
                results_dict[key] = value
    return results_dict

def generate_result_txt(results_dic):
    results = ""
    sorted_keys=sorted(results_dic.keys(), key=lambda x:x.lower())
    for key in sorted_keys:
        results += key + ": " + str(results_dic[key]) + "\n"
    return results


def is_float(val, print_val=False):
    try:
        float(val)
        return True
    except:
        if print_val:
            print(val)
        return False
