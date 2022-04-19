from enum import Enum

class ModelName(Enum):
    Empty = 0
    Standard = 1
    Hendrycks2020AugMix_WRN = 2
    ImageNetDeepAugmentAndAugmix = 3
    ScanPath_Simple = 4
    ResNet16 = 5
    ResNet2810 = 6
    Wong2020Fast = 7
    DenoisingAutoencoder = 8
    CIFAR100_ResNet32 = 9
    Standard_R50 = 10

    def __str__(self):
        return self.name

class DataSetName(Enum):
    cifar10 = 0
    imagenet = 1
    imagenet_a = 2
    mit1003 = 3
    cifar100 = 4
    cifar100Subset = 5
    imagenetSubset = 6
    kootstra = 7
    siena12 = 8
    toronto = 9

    def __str__(self):
        if self.value == 3:
            return "MIT1003"
        elif self.value == 7:
            return "Kootstra"
        elif self.value == 8:
            return "Siena12"
        elif self.value == 9:
            return "Toronto"
        else:
            return self.name

class Plots(Enum):
    plot_scanpaths = 0
    plot_scanpath_loss = 1
    plot_scanpath_diversity = 2
    plot_scanpath_diversity_grouped_by_labels = 3

class LossFunction(Enum):
    # Cross entropy
    ce = 0

    def __str__(self):
        return self.name

class DownStreamType(Enum):
    Classifier = 0
    Reconstructer = 1

    def __str__(self):
        if self.value == 0:
            return "CLASS"
        if self.value == 1:
            return "REC"
        return self.name
