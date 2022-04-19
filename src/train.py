import torch
import torch.nn as nn
import numpy as np
import os
import pickle

from src.foveation import get_foveation, calculate_blur, get_foveation_pixel_pos
from src.data import get_dataset_information, get_stimuli_img, is_stimuli_data, get_stimuli_img_sizes, get_image_net_label_from_idx, get_cifar10_label_from_idx
from src.models import load_scan_path_model
from src.plot import plot_foveation, plot_pipeline, plot_image
from torch.utils.tensorboard import SummaryWriter
from src.CustomEnums import DownStreamType, DataSetName
from src.SaveData import save_foveation, get_foveation_data_path
from src.AdversarialAttack import attack_downstream

def run_train(conf, train_loader, scan_path_model, downstream_model):
    model, loaded = load_scan_path_model(conf, scan_path_model, "Last")
    epochs = conf.epochs
    noise_factor = conf.noise_factor
    deblur_steps = conf.deblur_steps
    foveation_size = conf.foveation_size
    filter_size = conf.filter_size
    criterion = downstream_model.criterion
    data_set_info = get_dataset_information(conf.dataset)
    image_size = data_set_info["shape"][-1]

    optimizer = torch.optim.Adam(scan_path_model.parameters(), lr=conf.lr)
    writer = SummaryWriter(conf.tensorboard_log_path())

    best_train_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # monitor training loss
        train_loss = 0.0

        ###################
        # train the model #
        ###################
        for i, data in enumerate(train_loader):
            # _ stands in for labels, here
            # no need to flatten images
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            targets = downstream_model.get_targets(images, labels)
            B = images.shape[0]
            # blur images
            blur = calculate_blur(images, filter_size, noise_factor)
            blurring_mask = torch.ones((B, 1, image_size, image_size), device='cuda')
            foveation_mask = torch.zeros((B, 1, image_size, image_size), device='cuda')
            for step in range(deblur_steps):
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                blurring_mask = blurring_mask.detach()
                foveation_mask = foveation_mask.detach()
                # forgetting
                foveation_mask.data = foveation_mask.data * conf.forgetting
                # scan path model
                foveation_pos = scan_path_model(images + (blur * blurring_mask)) #Fix
                # Generate mask from position
                current_foveation_mask = get_foveation(foveation_pos, foveation_size, 1, image_size)
                # add to foveation mask
                foveation_mask = foveation_mask + current_foveation_mask
                # calcultate current mask
                blurring_mask = torch.clip((torch.ones((B, 1, image_size, image_size), device='cuda') - foveation_mask), 0, 1)
                # classifer
                output = downstream_model(images + (blur * blurring_mask))
                # calculate the loss
                loss = criterion(output, targets).mean()
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update running training loss
                train_loss += loss.item() * images.size(0)
                # print avg training statistics
                current_iteration = ((epoch - 1) * len(train_loader) * deblur_steps) + (i * deblur_steps) + step
                log(writer, current_iteration, loss.item(), "Batch")
                if i % 100 == 0:
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch":epoch, "iteration":i}, conf.model_save_path("Last", False))
                if i % 30 == 0 and conf.plot:
                    plot_foveation(conf, images, blur, blurring_mask, foveation_mask)
        train_loss = train_loss / len(train_loader) / deblur_steps
        log(writer, epoch, train_loss, "Epoch")
        if best_train_loss > train_loss:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, conf.model_save_path("Best", False))
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epochs, "iteration": 0}, conf.model_save_path("Last", False))

def run_input_optimize(conf, test_loader, downstream_model):
    opt_iterations = conf.epochs
    noise_factor = conf.noise_factor
    deblur_steps = conf.deblur_steps
    foveation_size = conf.foveation_size
    filter_size = conf.filter_size
    criterion = downstream_model.criterion

    dataset = conf.test_dataset
    data_set_info = get_dataset_information(conf.dataset)

    image_size = data_set_info["shape"][-1]

    label_func = get_image_net_label_from_idx if conf.dataset == DataSetName.imagenet else get_cifar10_label_from_idx

    suffix = "_adversarial" if conf.adversarial else ""
    foveation_data_path = get_foveation_data_path(conf, suffix=suffix)

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

    scanpath_dict = {}
    label_dict = {}
    loss_history = {}

    for i, data in enumerate(test_loader):
        # _ stands in for labels, here
        # no need to flatten images
        min, max = i * conf.batch_size, (i + 1) * conf.batch_size
        images, labels = data
        images, labels = images.cuda(), labels.cuda()

        if conf.downstream_model_type == DownStreamType.Classifier:
            output = downstream_model(images)
            labels = output.argmax(1)
            if conf.adversarial:
                delta = attack_downstream(conf, downstream_model, images, labels, 10)
                labels = downstream_model.predict_targets(images + delta)
            label_names = [[idx.item(), label_func(idx.item())] for idx in labels]
            label_names = np.array(label_names)
        else:
            labels = None
            label_names = None

        targets = downstream_model.get_targets(images, labels)
        B = images.shape[0]
        # blur images
        blur = calculate_blur(images, filter_size, noise_factor, conf.blur_sigma)
        foveation_mask = torch.zeros((B, 1, image_size, image_size), device='cuda')
        foveation_pixel_posisitions = []
        for step in range(deblur_steps):
            if len(loss_history) < deblur_steps:
                loss_history["step_" + str(step)] = []

            images.requires_grad_ = True

            best_foveation_pos = torch.zeros((images.shape[0], 2, 1, 1), device='cuda')  # * 2 - 1
            best_loss = torch.ones((images.shape[0]), device='cuda') * float("inf")
            foveation_pos = torch.zeros((images.shape[0], 2, 1, 1), device='cuda', requires_grad=True)
            #optimizer = torch.optim.SGD([foveation_pos], lr=conf.lr, momentum=0.9)

            for _ in range(opt_iterations):
                # Generate mask from position
                current_foveation_mask = get_foveation(foveation_pos, foveation_size, 1, image_size)
                temporary_foveation_mask = current_foveation_mask + foveation_mask
                # calcultate current mask
                blurring_mask = torch.clip((torch.ones((B, 1, image_size, image_size), device='cuda') - temporary_foveation_mask), 0, 1)
                # classifer
                output = downstream_model(images + (blur * blurring_mask))
                # calculate the loss
                loss = criterion(output, targets)
                total_loss = loss.mean()
                # backward pass: compute gradient of the loss with respect to model parameters
                total_loss.backward()
                # perform a single optimization step (parameter update)
                #optimizer.step()
                foveation_pos.data -= torch.sign(foveation_pos.grad) * conf.lr
                # clip positions
                foveation_pos.data = torch.clip(foveation_pos.data, -2.5, 2.5)
                # reset gradient
                #optimizer.zero_grad()
                foveation_pos.grad.zero_()
                #save best
                idxs = loss < best_loss
                best_loss[idxs] = loss[idxs]
                best_foveation_pos[idxs] = foveation_pos[idxs]
                if torch.sum(~idxs) > 0:
                    foveation_pos.data[~idxs] += torch.rand_like(best_foveation_pos.data[~idxs]) * conf.lr - conf.lr / 2

            # Update mask
            current_foveation_mask = get_foveation(best_foveation_pos, foveation_size, 1, image_size)
            foveation_mask = (foveation_mask * conf.forgetting + current_foveation_mask).detach()
            blurring_mask = torch.clip((torch.ones((B, 1, image_size, image_size), device='cuda') - foveation_mask), 0, 1)
            reconstructed = None
            if conf.downstream_model_type == DownStreamType.Reconstructer:
                reconstructed = downstream_model(images + (blur * blurring_mask))
            if i < 5:
                plot_foveation(conf, images, blur, blurring_mask, foveation_mask, f"{i}_{step}", ["Optimized" + suffix], reconstructed=reconstructed)
                plot_pipeline(conf, images, blur, blurring_mask, foveation_mask, f"{i}_{step}", ["Optimized" + suffix])
            pixel_pos = get_foveation_pixel_pos(best_foveation_pos, img_sizes[min:max], resize, crop, foveation_size, 1, image_size)
            foveation_pixel_posisitions.append(pixel_pos)
            #Loss history
            loss_history["step_" + str(step)].extend(list(loss.detach().cpu().numpy()))

        b_scanpath_dict, b_label_dict = save_foveation(conf, np.stack(foveation_pixel_posisitions, 1).squeeze(),
                                                       conf.downstream_model_type.name + "_Optimize", names=img_names[min:max], labels=label_names, suffix=suffix)
        scanpath_dict.update(b_scanpath_dict)
        label_dict.update(b_label_dict)

    with open(foveation_data_path + "loss_history.pickle", 'wb') as f:
        pickle.dump(loss_history, f)
    with open(foveation_data_path + "scanpath_data.pickle", 'wb') as f:
        pickle.dump(scanpath_dict, f)
    if len(label_dict) > 0:
        with open(foveation_data_path + "label_dict.pickle", 'wb') as f:
            pickle.dump(label_dict, f)

def run_input_optimize_sequence(conf, test_loader, downstream_model):
    opt_iterations = conf.epochs
    noise_factor = conf.noise_factor
    deblur_steps = conf.deblur_steps
    foveation_size = conf.foveation_size
    filter_size = conf.filter_size
    criterion = downstream_model.criterion
    data_set_info = get_dataset_information(conf.dataset)
    image_size = data_set_info["shape"][-1]

    dataset = conf.dataset
    if conf.test_dataset is not None:
        dataset = conf.test_dataset
    data_set_info = get_dataset_information(conf.dataset)
    image_size = data_set_info["shape"][-1]

    if dataset == is_stimuli_data(dataset):
        img_paths = np.array(test_loader.dataset.imgs)
        img_names = [img[0].split("\\")[-1][:-4] for img in img_paths]
        img_names = np.array(img_names)
        img_sizes = get_stimuli_img_sizes(conf, dataset)
        mit_1003_image = get_stimuli_img(conf, dataset)
        resize = test_loader.dataset.transform.transforms[0].size
        crop = test_loader.dataset.transform.transforms[1].size[0]

    for i, data in enumerate(test_loader):
        # _ stands in for labels, here
        # no need to flatten images
        min, max = i * conf.batch_size, (i + 1) * conf.batch_size
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        labels = downstream_model(images).argmax(1)
        targets = downstream_model.get_targets(images, labels)
        B = images.shape[0]
        # blur images
        blur = calculate_blur(images, filter_size, noise_factor, conf.blur_sigma)
        foveation_pixel_posisitions = []
        images.requires_grad_ = True

        best_foveation_pos = torch.zeros((deblur_steps, images.shape[0], 2, 1, 1), device='cuda')
        best_loss = torch.ones((deblur_steps, images.shape[0]), device='cuda') * float("inf")
        foveation_pos = torch.zeros((deblur_steps, images.shape[0], 2, 1, 1), device='cuda', requires_grad=True)
        # foveation_pos.data = foveation_pos.data + (torch.rand_like(foveation_pos) - 0.5).data
        # optimizer = torch.optim.SGD([foveation_pos], lr=conf.lr, momentum=0.9)
        loss_sequence = torch.ones((deblur_steps, images.shape[0]), device='cuda') * float("inf")
        for _ in range(opt_iterations):
            foveation_mask = torch.zeros((B, 1, image_size, image_size), device='cuda')
            for step in range(deblur_steps):
                # Generate mask from position
                current_foveation_mask = get_foveation(foveation_pos[step], foveation_size, 1, image_size)
                temporary_foveation_mask = current_foveation_mask + foveation_mask
                # calcultate current mask
                blurring_mask = torch.clip((torch.ones((B, 1, image_size, image_size), device='cuda') - temporary_foveation_mask), 0, 1)
                # classifer
                output = downstream_model(images + (blur * blurring_mask))
                # calculate the loss
                loss = criterion(output, targets)
                loss_sequence[step] = loss.detach()
                total_loss = loss.mean()
                # backward pass: compute gradient of the loss with respect to model parameters
                total_loss.backward(retain_graph=True) #retain_graph=True
                #update mask
                foveation_mask = (foveation_mask * conf.forgetting + current_foveation_mask)
            # perform a single optimization step (parameter update)
            foveation_pos.data -= torch.sign(foveation_pos.grad) * conf.lr # optimizer.step()
            # save best
            idxs = loss_sequence < best_loss
            best_loss[idxs] = loss_sequence[idxs]
            best_foveation_pos[idxs] = foveation_pos[idxs]
            if torch.sum(~idxs) > 0:
                foveation_pos.data[~idxs] += torch.rand_like(best_foveation_pos.data[~idxs]) * conf.lr - conf.lr / 2
            # reset gradient
            foveation_pos.grad.zero_() # optimizer.zero_grad()
        # Update mask
        foveation_mask = torch.zeros((B, 1, image_size, image_size), device='cuda')
        for step in range(deblur_steps):
            current_foveation_mask = get_foveation(best_foveation_pos[step], foveation_size, 1, image_size)
            foveation_mask = (foveation_mask * conf.forgetting + current_foveation_mask).detach()
            blurring_mask = torch.clip((torch.ones((B, 1, image_size, image_size), device='cuda') - foveation_mask), 0, 1)

            if dataset == DataSetName.mit1003:
                pixel_pos = get_foveation_pixel_pos(best_foveation_pos[step], img_sizes[min:max], resize, crop, foveation_size, 1, image_size)
                foveation_pixel_posisitions.append(pixel_pos)

            reconstructed = None
            if conf.downstream_model_type == DownStreamType.Reconstructer:
                output = downstream_model(images + (blur * blurring_mask))
                reconstructed = output

            if i < 5:
                plot_foveation(conf, images, blur, blurring_mask, foveation_mask, f"{i}_{step}", ["Optimized"], reconstructed=reconstructed)

        if dataset == DataSetName.mit1003:
            save_foveation(conf, np.stack(foveation_pixel_posisitions, 1).squeeze(), conf.downstream_model_type.name + "_Optimize", names=img_names[min:max])


def log(writer, iteration, train_loss, type):
    log_to_tensorboard(writer, iteration, {"Loss": train_loss}, type)

def log_to_tensorboard(writer, n_iter, log_dict, type):
    for key in log_dict:
        writer.add_scalar(type + "/" + key, log_dict[key], n_iter)