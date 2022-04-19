import torch
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms

def get_foveation(positions, foveation_size, foveation_aggregation, image_size):
    mask = calc_gaussian(foveation_aggregation, foveation_size, image_size, positions)
    return mask

def get_foveation_pixel_pos(positions, image_sizes, resize, crop, foveation_size, foveation_aggregation, image_size):
    mask = calc_gaussian(foveation_aggregation, foveation_size, image_size, positions)
    B = mask.shape[0]
    W = mask.shape[-1]
    idxs = mask.view(B, -1).argmax(1)
    x = (idxs % W).cpu().numpy()
    y = (W - (idxs // W)).cpu().numpy()

    smaller_image_size = np.min(image_sizes, 1)
    factor = smaller_image_size / resize
    new_image_sizes = image_sizes / factor.reshape(-1, 1)
    x_crop, y_crop = x + (new_image_sizes[:, 1] - crop) // 2, y + (new_image_sizes[:, 0] - crop) // 2
    x_rel, y_rel = x_crop / new_image_sizes[:, 1], y_crop / new_image_sizes[:, 0]
    x_final, y_final = x_rel * image_sizes[:, 1], y_rel * image_sizes[:, 0]
    x_final, y_final = np.clip(x_final, 0, image_sizes[:, 1]), np.clip(y_final, 0, image_sizes[:, 0])

    return np.stack((x_final, y_final), 1).astype(int)

def get_foveation_pos(positions, image_size):
    B = positions.shape[0]
    steps = positions.shape[1]
    positions_cuda = torch.tensor(positions, device='cuda').view(B, steps, 2, 1, 1)
    if len(image_size.shape) > 1:
        positions_cuda[:, :, 1] = image_size[:, 0:1] - positions_cuda[:, :, 1]
    else:
        positions_cuda[:, :, 1] = image_size - positions_cuda[:, :, 1]

    if len(image_size.shape) > 1:
       image_size = image_size.view(image_size.shape[0], 1, 2, 1, 1)
       tmp = image_size[:, :, 0].clone()
       image_size[:, :, 0] = image_size[:, :, 1]
       image_size[:, :, 1] = tmp

    positions_cuda = positions_cuda / (image_size / 2)
    positions_cuda -= 1
    tmp = positions_cuda[:, :, 0].clone()
    positions_cuda[:, :, 0] = positions_cuda[:, :, 1]
    positions_cuda[:, :, 1] = tmp
    return positions_cuda.type(torch.float32)

def calc_gaussian(a, std_dev, size, positions):
    B = positions.shape[0]
    xa, ya = create_grid(B, size)
    xa = xa - positions[:, 0]
    ya = ya - positions[:, 1]
    distance = (xa**2 + ya**2)
    g = a * torch.exp(-distance / std_dev)
    return g.view(B, 1, size, size)

def create_grid(batch_size, size):
    t = torch.linspace(-1, 1, size).cuda()
    xa, ya = torch.meshgrid([t, t])
    xa = xa.view(1, size, size).repeat(batch_size, 1, 1)
    ya = ya.view(1, size, size).repeat(batch_size, 1, 1)
    return xa, ya

def calculate_blur(images, filter_size, noise_factor, sigma=5):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(window_size, channel, sigma):
        _1D_window = gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    noisy_imgs = images + noise_factor * torch.randn(*images.shape, device='cuda')
    window = create_window(filter_size, 3, sigma).cuda()
    pad = nn.ReflectionPad2d(padding=filter_size // 2)
    noisy_imgs_pad = pad(noisy_imgs)
    blured_images = F.conv2d(noisy_imgs_pad, window, groups=3)
    # Clip the images to be between 0 and 1
    blured_images = torch.clip(blured_images, 0., 1.)
    blur = blured_images - images
    return blur