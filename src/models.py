import robustbench as rb
import torch.nn as nn
import torchvision.models as models
import torch
import os

from src.ResNet import WideResNet
from src.ResNet_CIFAR100 import cifar100_resnet32
from src.data import imageNetA_mask, get_dataset_information, get_mean_std
from src.CustomEnums import DataSetName, ModelName, DownStreamType


def get_scan_path_model(conf):
    data_set_info = get_dataset_information(conf.dataset)
    image_size = data_set_info["shape"][-1]
    if conf.scan_path_model == ModelName.ScanPath_Simple:
        model = scan_path_model_simpel(image_size, conf.foveation_size)
    elif conf.scan_path_model == ModelName.ResNet16:
        model = WideResNet(16, 2, 1)
    elif conf.scan_path_model == ModelName.ResNet2810:
        model = WideResNet(28, 2, 10)

    model = model.cuda()

    return model

def load_scan_path_model(conf, model, type):
    if conf.load:
        path = conf.model_save_path(type, False)
        if os.path.isfile(path):
            state_dict = torch.load(path)["model"]
            model.load_state_dict(state_dict)
            model = model.cuda()
            return model, True
        return model, False
    else:
        return model, False

def get_down_stream_model(conf, downstream_model=None, dataset=None):
    if downstream_model is None:
        downstream_model = conf.downstream_model
    if dataset is None:
        dataset = conf.dataset

    dir = conf.model_save_path("", True, dataset=dataset).replace("Subset", "")
    model_name = downstream_model.name.replace("_L2", "")

    if downstream_model == ModelName.ImageNetDeepAugmentAndAugmix:
        model = models.__dict__["resnet50"]()
        dir = dir.replace("mit1003", "imagenet")
        path = dir.replace("imagenet_a", "imagenet").replace("imagenet", "imagenet")
        path += model_name + ".pt"
        checkpoint = torch.load(path)
        state_dict = load_state_dict(checkpoint)
        model.load_state_dict(state_dict)
        model = ImageNeWrapper(model)
    elif downstream_model == ModelName.Hendrycks2020AugMix_WRN:
        model = rb.utils.load_model(model_name=model_name, dataset="cifar10", model_dir=dir, norm="corruptions")
    elif downstream_model == ModelName.Standard:
        model = rb.utils.load_model(model_name=model_name, dataset=conf.dataset.name, model_dir=dir)
    elif downstream_model == ModelName.CIFAR100_ResNet32:
        model = cifar100_resnet32()
        path = "/".join(dir.split("/")[:-2]) + "/data_set(cifar100)/" + model_name + ".pt"
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        mean_std = get_mean_std(DataSetName.cifar100)
        model = ClassifierWrapper(model, mean_std=mean_std)
    elif downstream_model == ModelName.Wong2020Fast:
        model = rb.utils.load_model(model_name=model_name, dataset="cifar10", model_dir=dir)
    elif downstream_model == ModelName.DenoisingAutoencoder:
        model = denoising_autoencoder(conf)
    elif downstream_model == ModelName.Standard_R50:
        model = models.resnet50(pretrained=True)
        model = ImageNeWrapper(model)

    if conf.downstream_model_type == DownStreamType.Classifier:
        model = ClassifierWrapper(model)
    elif conf.downstream_model_type == DownStreamType.Reconstructer:
        model = ReconstructionWrapper(model)

    model = model.cuda()
    return model

class ClassifierWrapper(nn.Module):
    def __init__(self, model, mean_std = None):
        super(ClassifierWrapper, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.mean_std = mean_std
    def forward(self, x):
        if self.mean_std is not None:
            x = (x - self.mean_std[0]) / self.mean_std[1]
        return self.model(x)

    def get_targets(self, x, y):
        return y

    def predict_targets(self, x):
        return self.model(x).argmax(1)

class ReconstructionWrapper(nn.Module):
    def __init__(self, model):
        super(ReconstructionWrapper, self).__init__()
        self.model = model
        self.mse = nn.MSELoss(reduction="none")
        def loss_func(x, y):
            loss = self.mse(x, y)
            loss = torch.mean(loss.view(loss.shape[0], -1), 1)
            return loss

        self.criterion = loss_func

    def forward(self, x):
        return self.model(x)

    def get_targets(self, x, y):
        return x

    def predict_targets(self, x):
        return self.model(x)

class ImageNetAWrapper(nn.Module):
    def __init__(self, model):
        super(ImageNetAWrapper, self).__init__()
        self.model = model
        self.mask = imageNetA_mask()

    def forward(self, x):
        y = self.model(x)
        y  = y[:, self.mask]
        return y

class ImageNeWrapper(nn.Module):
    def __init__(self, model):
        super(ImageNeWrapper, self).__init__()
        self.model = model
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()

    def forward(self, x):
        x = (x - self.mean) / self.std
        y = self.model(x)
        return y

class RobustnessWraper(nn.Module):
    def __init__(self, model):
        super(RobustnessWraper, self).__init__()
        self.model = model

    def forward(self, x):
        y, _ = self.model(x)
        return y

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Clip(nn.Module):

    def __init__(self, size):
        super(Clip, self).__init__()
        self.size = size

    def forward(self, x):
        return torch.clip(x, -self.size, self.size)

class denoising_autoencoder(nn.Module):
    def __init__(self, conf):
        super(denoising_autoencoder, self).__init__()
        path = conf.model_save_path("", conf.pretrained)
        self.denoiser = DnCNN()
        self.conf = conf
        if conf.pretrained:
            if conf.dataset == DataSetName.imagenet:
                state_dict = torch.load(path + 'denoiser/imagenet_denoise_DnCNN_last_sd.pt')
            else:
                state_dict = torch.load(path + 'denoiser/denoiser_cifar10.pt')
            self.denoiser.load_state_dict(state_dict)

    def forward(self, x):
        #if self.conf.dataset == DataSetName.imagenet:
        #    x = x * 255
        return self.denoiser(x)

class scan_path_model_simpel(nn.Module):
    def __init__(self, size, foveation_size):
        super(scan_path_model_simpel, self).__init__()

        self.backbone = []
        self.size = size
        self.foveation_size = foveation_size

        self.backbone.append(nn.Conv2d(3, 32, 4, stride=2, padding=1))
        self.backbone.append(nn.BatchNorm2d(32))
        self.backbone.append(nn.ReLU())

        self.backbone.append(nn.Conv2d(32, 64, 4, stride=2, padding=1))
        self.backbone.append(nn.ReLU())
        self.backbone.append(nn.BatchNorm2d(64))

        self.backbone.append(nn.Conv2d(64, 128, 4, stride=2, padding=1))
        self.backbone.append(nn.ReLU())
        self.backbone.append(nn.BatchNorm2d(128))

        self.backbone.append(Flatten())
        width = int(size / 8)
        self.backbone.append(nn.Linear(128 * width * width, 2))
        self.backbone.append(Clip(2.5))
        self.backbone = nn.Sequential(*self.backbone)

    def forward(self, x):
        pos = self.backbone(x).view(-1, 2, 1, 1)
        return pos


class DnCNN(nn.Module):

    def __init__(self, num_layers=17, num_features=64):
        super(DnCNN, self).__init__()
        layers = [nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(num_features, 3, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        y = inputs
        residual = self.layers(y)
        return y - residual

def load_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint["state_dict"]
        keys = list(checkpoint.keys())
        for key in keys:
            checkpoint[key.replace("module.", "")] = checkpoint[key]
            del checkpoint[key]
        return checkpoint