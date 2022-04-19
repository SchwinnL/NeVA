import torch
from src.foveation import calculate_blur

def attack_scanpath(conf, model, X, attack_iters):
    attack_norm = 4/255
    delta = torch.zeros_like(X).requires_grad_()
    for _ in range(attack_iters):
        blur = calculate_blur(X + delta, conf.filter_size, conf.noise_factor)
        foveation_pos_adversarial = model(X + delta + (blur))
        loss = torch.abs((torch.ones_like(foveation_pos_adversarial) - foveation_pos_adversarial)).mean()
        grad = torch.autograd.grad(loss, delta)[0].detach()
        delta.data = delta.data - (torch.sign(grad) * attack_norm / 4)
        delta.data = torch.clamp(delta.data, -attack_norm, attack_norm)
    return delta

def attack_downstream(conf, model, X, y, attack_iters):
    attack_norm = 4/255
    delta = torch.zeros_like(X).requires_grad_()
    for _ in range(attack_iters):
        output = model(X + delta)
        loss = model.criterion(output, y).mean()
        grad = torch.autograd.grad(loss, delta)[0].detach()
        delta.data = delta.data + (torch.sign(grad) * attack_norm / 4)
        delta.data = torch.clamp(delta.data, -attack_norm, attack_norm)
    return delta
