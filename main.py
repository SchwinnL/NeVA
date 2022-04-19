from src.Configuration import Conf
from src.Experiments import train_experiment, test_experiment, accuracy_experiment, generate_scanpath_adversarial_experiment, standard_accuracy_experiment, plot_experiment, optimize_foveation_position_experiment, optimize_sequence_experiment
import itertools
from read_yaml import open_yaml
import argparse

experiment_path = "C:/Users/leosc/Experiments/UVA/"
data_path = "C:/Users/leosc/Datasets/"

exp1 = "SubsetScanpaths/CIFAR10_CIFAR100Subset_Classifier_Test"
exp2 = "Classifier/CIFAR10_MIT1003_Optimized"

exp3 = "SubsetScanpaths/CIFAR10_ImageNetSubset_Autoencoder_Optimized"
exp4 = "AccuracyFoveated/CIFAR10_ImageNetSubset_Classifier_Optimized"

exp5 = "SubsetScanpaths/CIFAR10_CIFAR100Subset_Autoencoder_Optimized"
exp6 = "AccuracyFoveated/CIFAR10_CIFAR100Subset_Autoencoder_Optimized"

exp7 = "TrainScanPaths/CIFAR10_Train_Classifier"

exp8 = "SubsetScanpaths/CIFAR10_MIT1003_Classifier_Optimized"

exp_robust = "TrainScanPaths/CIFAR10_Train_Classifier_Wong"

exp_plot_1 = "Plot/CIFAR10_MIT1003_Classifier_Optimized"
exp_plot_2 = "Plot/CIFAR10_MIT1003_Autoencoder_Optimized"
exp_plot_3 = "Plot/CIFAR10_MIT1003_Classifier_Trained"
exp_plot_4 = "Plot/ImageNet_MIT1003_Classifier_Optimized"
exp_plot_5 = "Plot/plot_scanpath_diversity"
exp_plot_6 = "Plot/plot_scanpath_diversity_grouped_by_labels"
exp_plot_7 = "Plot/plot_scanpath_diversity_grouped_by_labels_imagenet"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", default=experiment_path)
    parser.add_argument("--data_path", default=data_path)
    parser.add_argument("--yaml_file", default=[exp_plot_5], nargs='*')
    return parser.parse_args()

def main():
    input_args = get_args()
    for yaml_file in input_args.yaml_file:
        print("Yaml File:", yaml_file)
        arguments, keys = open_yaml(yaml_file)
        runs = list(itertools.product(*arguments))
        for run in runs:
            current_args = {}
            current_args["experiment_path"] = input_args.experiment_path
            current_args["data_path"] = input_args.data_path
            for i in range(len(keys)):
                current_args[keys[i]] = run[i]
            conf = Conf(current_args)
            if conf.train:
                train_experiment(conf)
            if conf.optimize_foveation_pos:
                optimize_foveation_position_experiment(conf)
            if conf.opt_sequence:
                optimize_sequence_experiment(conf)
            if conf.test:
                test_experiment(conf)
            if conf.generate_scanpath_adversarial:
                generate_scanpath_adversarial_experiment(conf)
            if conf.accuracy:
                accuracy_experiment(conf)
            if conf.standard_accuracy:
                standard_accuracy_experiment(conf)
            if conf.plot and conf.plot_type is not None:
                plot_experiment(conf)

if __name__ == "__main__":
    main()