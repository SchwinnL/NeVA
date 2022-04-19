import torch
import os
import numpy as np
import re
import pandas as pd
from .CustomEnums import DownStreamType, ModelName

def save(conf, foveation_pos, batch_number, batch_size, deblur_step, sub_dir, names=None):
    data_path = conf.data_path
    p = [data_path, "UVA/", sub_dir + "/", conf.full_model_name() + "/"]
    conf.create_paths(p)
    path = "".join(p)
    suffix = f"{batch_number}_{batch_number * batch_size}_{(batch_number + 1) * batch_size}_{deblur_step}_{conf.deblur_steps}"

    #torch.save(foveation_pos.detach(), path + suffix)

def save_foveation(conf, foveation_positions, sub_dir, names, labels=None, test_forgetting=None, suffix=""):

    path = get_foveation_data_path(conf, test_forgetting, suffix)

    scanpath_dict = {}
    label_dict = {}

    for i, name in enumerate(names):
        positions = foveation_positions[i]
        zeros = np.zeros((positions.shape[0], 2))
        positions = np.concatenate((positions, zeros), 1)
        scanpath_dict[name.split("/")[-1]] = positions

        if labels is not None:
            label_dict[name.split("/")[-1]] = f"{labels[i, 0]}: {labels[i, 1]}"
    return scanpath_dict, label_dict
    #torch.save(foveation_pos.detach(), path + suffix)

def get_foveation_data_path(conf, test_forgetting=None, suffix=""):
    data_path = conf.data_path

    model_name = ""
    if conf.dataset is not None:
        model_name += conf.dataset.name
    if conf.downstream_model_type is not None:
        model_name += "_" + str(conf.downstream_model_type)

    if conf.method_name is None:
        model_name += "_PRETRAIN"

    if conf.trained_scanpath_model and conf.trained_scanpath_model is not None:
        model_name += "_TRAIN"
    elif conf.optimize_foveation_pos and conf.optimize_foveation_pos is not None or conf.optimization is not None and conf.optimization:
        model_name += "_OPT"
    elif conf.opt_sequence and conf.opt_sequence is not None:
        model_name += "_OPTSEQ"

    if conf.forgetting is not None:
        model_name += "_" + str(conf.forgetting).replace(".", "")
    if test_forgetting is not None:
        model_name += "_tf" + str(test_forgetting).replace(".", "")

    if conf.method_name is not None:
        model_name += f"_{conf.method_name}"

    if conf.downstream_model not in [ModelName.Standard, ModelName.DenoisingAutoencoder, ModelName.ImageNetDeepAugmentAndAugmix]:
        model_name = conf.downstream_model.name + "_" + model_name

    p = [data_path, "UVA/", f"Scan_Paths{suffix}/", model_name + "/", conf.test_dataset.name + "/"]
    conf.create_paths(p)
    path = "".join(p)
    return path