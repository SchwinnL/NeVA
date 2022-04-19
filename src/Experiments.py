from src.models import get_down_stream_model, get_scan_path_model, load_scan_path_model
from src.data import get_data_set
from src.train import run_train, run_input_optimize, run_input_optimize_sequence
from .CustomEnums import ModelName, DataSetName, Plots
from src.Test import run_scanpath, run_accuracy, run_accuracy_foveated_clean, run_scanpath_adversarial
from src.plot import plot_from_foveation_position, plot_scanpath_loss, plot_scanpath_diversity, plot_scanpath_diversity_grouped_by_labels

import torch
import os

def train_experiment(conf):
    down_stream_model = get_down_stream_model(conf)
    scan_path_model = get_scan_path_model(conf)

    train_loader, test_loader = get_data_set(conf, True)
    run_train(conf, train_loader, scan_path_model, down_stream_model)

def optimize_foveation_position_experiment(conf):
    down_stream_model = get_down_stream_model(conf)

    if conf.dataset == DataSetName.cifar10 and conf.test_dataset == DataSetName.imagenetSubset:
        resize = True
    else:
        resize = False

    _, test_loader = get_data_set(conf, False, resize=resize)
    run_input_optimize(conf, test_loader, down_stream_model)

def optimize_sequence_experiment(conf):
    down_stream_model = get_down_stream_model(conf)

    _, test_loader = get_data_set(conf, False)
    run_input_optimize_sequence(conf, test_loader, down_stream_model)

def test_experiment(conf):
    down_stream_model = get_down_stream_model(conf)
    scan_path_model = get_scan_path_model(conf)
    scan_path_model, loaded = load_scan_path_model(conf, scan_path_model, "Best")
    if not loaded:
        print(f"could not find model: {conf.scan_path_model.name}")
        return
    if conf.test_dataset == None:
        raise Exception("Define SubsetScanpaths Dataset!")
    _, test_loader = get_data_set(conf, False, conf.test_dataset, resize=True)

    run_scanpath(conf, down_stream_model, scan_path_model, test_loader)

def generate_scanpath_adversarial_experiment(conf):
    down_stream_model = get_down_stream_model(conf)
    scan_path_model = get_scan_path_model(conf)
    scan_path_model, loaded = load_scan_path_model(conf, scan_path_model, "Best")
    if not loaded:
        print(f"could not find model: {conf.scan_path_model.name}")
        return
    if conf.test_dataset == None:
        raise Exception("Define SubsetScanpaths Dataset!")
    _, test_loader = get_data_set(conf, False, conf.test_dataset, resize=True)

    run_scanpath_adversarial(conf, down_stream_model, scan_path_model, test_loader)

def accuracy_experiment(conf):
    down_stream_model = get_down_stream_model(conf, conf.downstream_model_test, conf.test_dataset)
    if conf.test_dataset == None:
        raise Exception("Define SubsetScanpaths Dataset!")
    _, test_loader = get_data_set(conf, False, conf.test_dataset)

    run_accuracy(conf, down_stream_model, test_loader)

def standard_accuracy_experiment(conf):
    down_stream_model = get_down_stream_model(conf, conf.downstream_model_test, conf.test_dataset)
    _, test_loader = get_data_set(conf, False)
    run_accuracy_foveated_clean(conf, down_stream_model, test_loader)

def plot_experiment(conf):
    #down_stream_model = get_down_stream_model(conf)
    if conf.plot_type == Plots.plot_scanpath_loss:
        plot_scanpath_loss(conf)
    elif conf.plot_type == Plots.plot_scanpaths:
        _, test_loader = get_data_set(conf, False)
        plot_from_foveation_position(conf, test_loader)
    elif conf.plot_type == Plots.plot_scanpath_diversity:
        plot_scanpath_diversity(conf)
    elif conf.plot_type == Plots.plot_scanpath_diversity_grouped_by_labels:
        plot_scanpath_diversity_grouped_by_labels(conf)