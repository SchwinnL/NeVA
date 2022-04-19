from src.CustomEnums import *

train_keys = ['scan_path_model', "downstream_model", 'dataset', 'foveation_size', 'forgetting', 'noise_factor', 'deblur_steps',
              'batch_size', 'epochs', 'seed', "downstream_model_type", "lr", "filter_size", "optimize_foveation_pos",
              "pretrained", 'blur_sigma', 'opt_sequence', 'method_name']
test_keys = ["plot_scripts", "train", "load", "test", "accuracy", "experiment_path", "data_path", 'plot_type', 'resize', 'crop', 'plot_img_names',
             'table_name', 'viable_conf_keys', 'table_row', 'table_col', 'corner_value', "standard_accuracy", "generate_scanpath_adversarial", 'adversarial',
             'highlight_results', 'plot', 'test_dataset', 'downstream_model_test', 'trained_scanpath_model', 'forgetting_test', 'test_batch_size', 'optimization',
             'scan_path_experiments']
optional = ["seed"]
optional_test = ["result_path"]

all_enums = {ModelName, DataSetName, LossFunction, Plots, DownStreamType}

enums = {"scan_path_model", "downstream_model", "downstream_model_test", 'plot_type', "dataset", "loss_function", 'plots', "downstream_model_type", "test_dataset"}