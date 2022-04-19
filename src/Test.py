import time

import torch
import torch.optim
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import pickle

from src.AdversarialAttack import attack_downstream
from src.plot import plot_foveation, plot_foveation_position, plot_pipeline
from src.CustomEnums import DataSetName, DownStreamType
from src.foveation import calculate_blur, get_foveation, get_foveation_pixel_pos, get_foveation_pos
from src.data import get_dataset_information, get_stimuli_img, get_stimuli_img_sizes, get_image_net_label_from_idx, get_cifar10_label_from_idx, is_stimuli_data
from src.SaveResults import save_result_dict
from src.SaveData import save_foveation, get_foveation_data_path

def run_scanpath(conf, downstream_model, scan_path_model, test_loader):
    print("Testing: {}".format(conf.scan_path_model.name))

    downstream_model.eval()
    scan_path_model.eval()

    start_time = time.time()

    dataset = conf.test_dataset
    data_set_info = get_dataset_information(conf.dataset)
    image_size = data_set_info["shape"][-1]

    label_func = get_image_net_label_from_idx if conf.dataset == DataSetName.imagenet else get_cifar10_label_from_idx
    forgetting_test = conf.forgetting_test if conf.forgetting_test is not None else [conf.forgetting]

    suffix = "_adversarial_downstream" if conf.adversarial else ""

    if is_stimuli_data(dataset):
        img_paths = np.array(test_loader.dataset.imgs)
        img_names = [img[0].split("\\")[-1][:-4] for img in img_paths]
        img_names = np.array(img_names)
        img_sizes = get_stimuli_img_sizes(conf, dataset)
        mit_1003_image = get_stimuli_img(conf, dataset)
        resize = test_loader.dataset.transform.transforms[0].size
        crop = test_loader.dataset.transform.transforms[1].size[0]
    else:
        size = len(test_loader.dataset.tensors[0])
        img_sizes = np.ones((size, 2)) * image_size
        crop = image_size
        resize = image_size
        img_names = np.arange(0, size).astype(str)

    for forgetting in forgetting_test:
        scanpath_dict = {}
        label_dict = {}
        loss_history = {}

        test_forgetting = None
        if len(forgetting_test) > 1:
            test_forgetting = forgetting

        foveation_data_path = get_foveation_data_path(conf, test_forgetting, suffix)
        if os.path.isfile(foveation_data_path + "scanpath_data.pickle"):
            print("Already scanpathas at:", foveation_data_path)
            continue

        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            B = X.shape[0]
            min, max = i*test_loader.batch_size,(i+1)*test_loader.batch_size
            targets = downstream_model.get_targets(X, y)
            if conf.downstream_model_type == DownStreamType.Classifier and conf.downstream_model_test is None:
                output = downstream_model(X)
                labels = output.argmax(1)
                label_names = [[idx.item(), label_func(idx.item())] for idx in labels]
                label_names = np.array(label_names)
            else:
                label_names = None
            # blur images
            blur = calculate_blur(X, conf.filter_size, conf.noise_factor)
            foveation_mask = torch.zeros((B, 1, image_size, image_size), device='cuda')
            blurring_mask = torch.ones((B, 1, image_size, image_size), device='cuda')
            with torch.no_grad():
                foveation_positions = []
                for step in range(10): #conf.deblur_steps
                    if len(loss_history) < 10:
                        loss_history["step_" + str(step)] = []

                    foveation_pos = scan_path_model(X + (blur * blurring_mask))
                    # Generate mask from position
                    current_foveation_mask = get_foveation(foveation_pos, conf.foveation_size, 1, image_size)
                    foveation_pos_pixel = get_foveation_pixel_pos(foveation_pos, img_sizes[min:max], resize, crop, conf.foveation_size, 1, image_size)
                    foveation_positions.append(foveation_pos_pixel)
                    foveation_mask = (foveation_mask * forgetting) + current_foveation_mask
                    blurring_mask = torch.ones((B, 1, image_size, image_size), device='cuda') - foveation_mask
                    blurring_mask = torch.clip(blurring_mask, 0, 1)
                    #if i < 5:
                    #    plot_foveation(conf, X, blur, blurring_mask, foveation_mask, f"{i}_{step}", ["Trained" + suffix])
                    #    plot_pipeline(conf, X, blur, blurring_mask, foveation_mask, f"{i}_{step}", ["Trained" + suffix])

                #b_scanpath_dict, b_label_dict = save_foveation(conf, np.stack(foveation_positions, 1).squeeze(), conf.downstream_model_type.name + "_Train",
                #                                               names=img_names[min:max], labels=label_names, test_forgetting=test_forgetting)
                #scanpath_dict.update(b_scanpath_dict)
                #label_dict.update(b_label_dict)

        with open(foveation_data_path + "scanpath_data.pickle", 'wb') as f:
            pickle.dump(scanpath_dict, f)
        if len(label_dict) > 0:
            with open(foveation_data_path + "label_dict.pickle", 'wb') as f:
                pickle.dump(label_dict, f)

    test_time = time.time() - start_time
    print(test_time)


def run_scanpath_adversarial(conf, downstream_model, scan_path_model, test_loader):
    print("Testing: {}".format(conf.scan_path_model.name))

    downstream_model.eval()
    scan_path_model.eval()

    start_time = time.time()

    dataset = conf.test_dataset
    data_set_info = get_dataset_information(conf.dataset)
    image_size = data_set_info["shape"][-1]

    label_func = get_image_net_label_from_idx if conf.dataset == DataSetName.imagenet else get_cifar10_label_from_idx
    forgetting_test = conf.forgetting_test if conf.forgetting_test is not None else [conf.forgetting]

    attack_iters = 20
    attack_norm = 4/255

    if is_stimuli_data(dataset):
        img_paths = np.array(test_loader.dataset.imgs)
        img_names = [img[0].split("\\")[-1][:-4] for img in img_paths]
        img_names = np.array(img_names)
        img_sizes = get_stimuli_img_sizes(conf, dataset)
        mit_1003_image = get_stimuli_img(conf, dataset)
        resize = test_loader.dataset.transform.transforms[0].size
        crop = test_loader.dataset.transform.transforms[1].size[0]
    else:
        size = len(test_loader.dataset.tensors[0])
        img_sizes = np.ones((size, 2)) * image_size
        crop = image_size
        resize = image_size
        img_names = np.arange(0, size).astype(str)

    for forgetting in forgetting_test:
        scanpath_dict = {}
        label_dict = {}
        loss_history = {}

        test_forgetting = None
        if len(forgetting_test) > 1:
            test_forgetting = forgetting

        foveation_data_path = get_foveation_data_path(conf, test_forgetting, suffix="_adversarial")
        if os.path.isfile(foveation_data_path + "scanpath_data.pickle"):
            print("Already scanpathas at:", foveation_data_path)
            continue

        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            delta = torch.zeros_like(X).requires_grad_()


            B = X.shape[0]
            min, max = i*test_loader.batch_size,(i+1)*test_loader.batch_size
            targets = downstream_model.get_targets(X, y)
            if conf.downstream_model_type == DownStreamType.Classifier and conf.downstream_model_test is None:
                output = downstream_model(X)
                labels = output.argmax(1)
                label_names = [[idx.item(), label_func(idx.item())] for idx in labels]
                label_names = np.array(label_names)
            else:
                label_names = None
            # blur images
            blur = calculate_blur(X, conf.filter_size, conf.noise_factor)
            foveation_mask = torch.zeros((B, 1, image_size, image_size), device='cuda')
            blurring_mask = torch.ones((B, 1, image_size, image_size), device='cuda')

            for _ in range(attack_iters):
                blur_ = calculate_blur(X + delta, conf.filter_size, conf.noise_factor)
                foveation_pos_adversarial = scan_path_model(X + delta + (blur_ * blurring_mask))
                loss = torch.abs((torch.ones_like(foveation_pos_adversarial) - foveation_pos_adversarial)).mean()
                grad = torch.autograd.grad(loss, delta)[0]
                delta.data = delta.data - (torch.sign(grad) * attack_norm / 4)
                delta.data = torch.clamp(delta.data, -attack_norm, attack_norm)

            X = X + delta

            with torch.no_grad():
                foveation_positions = []
                for step in range(10): #conf.deblur_steps
                    if len(loss_history) < 10:
                        loss_history["step_" + str(step)] = []

                    foveation_pos = scan_path_model(X + (blur * blurring_mask))
                    # Generate mask from position
                    current_foveation_mask = get_foveation(foveation_pos, conf.foveation_size, 1, image_size)
                    foveation_pos_pixel = get_foveation_pixel_pos(foveation_pos, img_sizes[min:max], resize, crop, conf.foveation_size, 1, image_size)
                    foveation_positions.append(foveation_pos_pixel)
                    foveation_mask = (foveation_mask * conf.forgetting) + current_foveation_mask
                    blurring_mask = torch.ones((B, 1, image_size, image_size), device='cuda') - foveation_mask
                    blurring_mask = torch.clip(blurring_mask, 0, 1)

                    #output = downstream_model(X + blur * blurring_mask)
                    #loss = downstream_model.criterion(output, targets)
                    #loss_history["step_" + str(step)].extend(list(loss.detach().cpu().numpy()))
                b_scanpath_dict, b_label_dict = save_foveation(conf, np.stack(foveation_positions, 1).squeeze(), conf.downstream_model_type.name + "_Train",
                                                               names=img_names[min:max], labels=label_names, test_forgetting=test_forgetting)
                scanpath_dict.update(b_scanpath_dict)
                label_dict.update(b_label_dict)

        #with open(foveation_data_path + "loss_history.pickle", 'wb') as f:
        #    pickle.dump(loss_history, f)
        with open(foveation_data_path + "scanpath_data.pickle", 'wb') as f:
            pickle.dump(scanpath_dict, f)
        if len(label_dict) > 0:
            with open(foveation_data_path + "label_dict.pickle", 'wb') as f:
                pickle.dump(label_dict, f)

    test_time = time.time() - start_time
    print(test_time)


def run_accuracy(conf, downstream_model, test_loader):
    print(f"Testing - Downstreammodel: {conf.downstream_model_test.name}")

    dataset = conf.test_dataset
    data_set_info = get_dataset_information(dataset)
    image_size = data_set_info["shape"][-1]

    if conf.method_name is not None:
        foveation_image_size = 224
        print(f"Testing Method: {conf.method_name}")
    elif conf.test_dataset == DataSetName.imagenetSubset and conf.dataset == DataSetName.cifar10:
        foveation_image_size = 32
    else:
        foveation_image_size = image_size

    suffix = "_adversarial" if conf.adversarial else ""

    if conf.method_name != "random":
        foveation_data_path = get_foveation_data_path(conf, suffix=suffix) + "scanpath_data.pickle"
        positions = pickle.load(open(foveation_data_path, 'rb'))
        keys = list(positions.keys())
        positions = np.array(list(positions.values()))[:, :, :2]
        foveation_pos = get_foveation_pos(positions, foveation_image_size)
    else:
        foveation_pos = torch.rand((1000, 10, 2, 1, 1), device='cuda') * 2 - 1

    downstream_model.eval()
    start_time = time.time()

    n = 0
    total_acc = 0
    sigma = conf.blur_sigma if conf.blur_sigma is not None else 5

    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        B = X.shape[0]
        min, max = i * test_loader.batch_size, (i + 1) * test_loader.batch_size
        # blur images
        blur = calculate_blur(X, conf.filter_size, conf.noise_factor, sigma=sigma)
        #blur = calculate_blur(X, 41, 0.15, 10)
        foveation_mask = torch.zeros((B, 1, image_size, image_size), device='cuda')
        blurring_mask = torch.ones((B, 1, image_size, image_size), device='cuda')
        with torch.no_grad():
            for step in range(10): #conf.deblur_steps
                # Generate mask from position
                current_foveation_mask = get_foveation(foveation_pos[min:max, step], conf.foveation_size, 1, image_size)
                foveation_mask = foveation_mask + current_foveation_mask
                blurring_mask = torch.ones((B, 1, image_size, image_size), device='cuda') - foveation_mask
                blurring_mask = torch.clip(blurring_mask, 0, 1)
                output = downstream_model(X + blurring_mask * blur)
                acc = (output.argmax(1) == y).sum().item()
                total_acc += acc
                n += y.size(0)
    test_time = time.time() - start_time
    print(test_time)

    result_dict = {}
    suffix = ""
    if conf.downstream_model_test is not None:
        suffix = "_" + conf.downstream_model_test.name

    result_dict[dataset.name + " accuracy"] = (total_acc / n) * 100
    save_result_dict(conf, result_dict, "accuracy" + suffix)

def run_accuracy_foveated_clean(conf, downstream_model, test_loader):
    print(f"Accuracy - Downstreammodel: {conf.downstream_model.name}")

    dataset = conf.test_dataset

    downstream_model.eval()
    start_time = time.time()

    n = 0
    total_acc = 0
    total_acc_blurred = 0
    sigma = conf.blur_sigma if conf.blur_sigma is not None else 5

    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        B = X.shape[0]
        # blur images
        blur = calculate_blur(X, conf.filter_size, conf.noise_factor, sigma=sigma)
        with torch.no_grad():
            for step in range(10): #conf.deblur_steps
                # Generate mask from position
                output_blurred = downstream_model(X + blur)
                acc_blurred = (output_blurred.argmax(1) == y).sum().item()

                output = downstream_model(X)
                acc = (output.argmax(1) == y).sum().item()

                total_acc_blurred += acc_blurred
                total_acc += acc
                n += y.size(0)

    test_time = time.time() - start_time
    print(test_time)

    result_dict = {}
    suffix = ""
    if conf.downstream_model_test is not None:
        suffix = "_" + conf.downstream_model_test.name

    result_dict[dataset.name + " accuracy clean"] = (total_acc / n) * 100
    result_dict[dataset.name + " accuracy blurred"] = (total_acc / n) * 100
    save_result_dict(conf, result_dict, "accuracy_clean_blurred" + suffix)