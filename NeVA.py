import torch
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class NeVAWrapper(nn.Module):
    def __init__(self, downstream_model, criterion, target_function, image_size, foveation_sigma, blur_filter_size, blur_sigma, forgetting, foveation_aggregation=1, device="cuda", data_min=0, data_max=1):
        super(NeVAWrapper, self).__init__()

        self.image_size = image_size
        self.blur_filter_size = blur_filter_size
        self.blur_sigma = blur_sigma
        self.foveation_sigma = foveation_sigma
        self.forgetting = forgetting
        self.foveation_aggregation = foveation_aggregation

        self.internal_representation = None
        self.ones = None
        self.device = device
        self.data_min = data_min
        self.data_max = data_max

        self.downstream_model = downstream_model
        self.criterion = criterion
        self.target_function = target_function

    def forward(self, x, foveation_positions):
        if self.internal_representation is None:
            raise Exception("First set internal representation with function: initialize_scanpath_generation()")
        foveation_area = get_foveation(self.foveation_aggregation, self.foveation_sigma, self.image_size, foveation_positions)
        current_foveation_area = self.internal_representation + foveation_area
        blurring_mask = torch.clip(self.ones - current_foveation_area, self.data_min, self.data_max)
        applied_blur = self.blur * blurring_mask

        output = self.downstream_model(x + applied_blur)

        return output

    def initialize_scanpath_generation(self, x, batch_size):
        self.internal_representation = torch.zeros((batch_size, 1, self.image_size, self.image_size), device='cuda')
        self.ones = torch.ones((batch_size, 1, self.image_size, self.image_size), device=self.device)
        self.blur = calculate_blur(x, self.blur_filter_size, self.blur_sigma)

    def run_optimization(self, x, labels, scanpath_length, opt_iterations, learning_rate):
        batch_size = x.size(0)
        targets = self.target_function(x, labels)
        self.initialize_scanpath_generation(x, batch_size)

        scanpath = []
        loss_history = []

        for step in range(scanpath_length):
            foveation_pos = torch.zeros((batch_size, 2, 1, 1), device='cuda', requires_grad=True)
            best_foveation_pos = torch.zeros((batch_size, 2, 1, 1), device='cuda')  # * 2 - 1
            best_loss = torch.ones((batch_size), device='cuda', dtype=torch.float16) * float("inf")

            for _ in range(opt_iterations):
                output = self(x, foveation_pos)
                # calculate the loss
                loss = self.criterion(output, targets)
                total_loss = loss.mean()
                # backward pass: compute gradient of the loss with respect to model parameters
                grad = torch.autograd.grad(total_loss, foveation_pos)[0]
                # perform a single optimization step (parameter update)
                foveation_pos.data -= torch.sign(grad) * learning_rate
                # save best
                idxs = loss < best_loss
                best_loss[idxs] = loss[idxs]
                best_foveation_pos[idxs] = foveation_pos[idxs]
                if torch.sum(~idxs) > 0:
                    #Jitter positions that are worse than before
                    foveation_pos.data[~idxs] += torch.rand_like(best_foveation_pos.data[~idxs]) * learning_rate - learning_rate / 2

            # Update internal representation
            current_foveation_mask = get_foveation(self.foveation_aggregation, self.foveation_sigma, self.image_size, best_foveation_pos)
            self.internal_representation = (self.internal_representation * self.forgetting + current_foveation_mask).detach()
            # Save positions in array
            scanpath.append(best_foveation_pos.detach().cpu().numpy())
            # Loss history
            loss_history.append(list(loss.detach().cpu().numpy()))
        return np.stack(scanpath, 1).squeeze(), np.stack(loss_history, 1).squeeze()

def calc_gaussian(a, std_dev, image_size, positions):
    B = positions.shape[0]
    xa, ya = create_grid(B, image_size)
    xa = xa - positions[:, 0]
    ya = ya - positions[:, 1]
    distance = (xa**2 + ya**2)
    g = a * torch.exp(-distance / std_dev)
    return g.view(B, 1, image_size, image_size)

def create_grid(batch_size, size):
    t = torch.linspace(-1, 1, size).cuda()
    xa, ya = torch.meshgrid([t, t])
    xa = xa.view(1, size, size).repeat(batch_size, 1, 1)
    ya = ya.view(1, size, size).repeat(batch_size, 1, 1)
    return xa, ya

def calculate_blur(images, blur_filter_size, sigma=5):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(window_size, channel, sigma):
        _1D_window = gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    window = create_window(blur_filter_size, 3, sigma).cuda()
    pad = nn.ReflectionPad2d(padding=blur_filter_size // 2)
    noisy_imgs_pad = pad(images)
    blured_images = F.conv2d(noisy_imgs_pad, window, groups=3)
    # Clip the images to be between 0 and 1
    blured_images = torch.clip(blured_images, 0., 1.)
    blur = blured_images - images
    return blur

def get_foveation(foveation_aggregation, foveation_sigma, image_size, positions):
    mask = calc_gaussian(foveation_aggregation, foveation_sigma, image_size, positions)
    return mask