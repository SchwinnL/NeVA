import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import robustbench as rb

from NeVA import NeVAWrapper

data_path = "Datasets/"
model_path = "Models/"

batch_size = 32
image_size = 224
lr = 0.1
optimization_steps = 20

scanpath_length = 10
foveation_sigma = 0.15
blur_filter_size = 5
forgetting = 0.1
blur_sigma = 5
criterion = torch.nn.CrossEntropyLoss(reduction="none")
def target_function(x, y):
    return y

# Load Dataset
transform = transforms.Compose([transforms.ToTensor()])
test = datasets.CIFAR10(data_path + "CIFAR10-data", train=False, download=True, transform=transform)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True)

# Load Model
model = rb.utils.load_model(model_name="Standard", dataset="cifar10", model_dir=model_path).cuda()

# Create NeVA Model
NeVA_model = NeVAWrapper(downstream_model=model,
                    criterion=criterion,
                    target_function=target_function,
                    image_size=image_size,
                    foveation_sigma=foveation_sigma,
                    blur_filter_size=blur_filter_size,
                    blur_sigma=blur_sigma,
                    forgetting=forgetting,
                    foveation_aggregation=1,
                    device='cuda')

scanpaths = []
loss_history = []

for i, data in enumerate(test_loader):
    images, _ = data
    images = images.cuda()

    output = model(images)
    pred_labels = output.argmax(1)

    current_scanpaths, current_loss_history = NeVA_model.run_optimization(images, pred_labels, scanpath_length, optimization_steps, lr)
    scanpaths.extend(current_scanpaths)
    loss_history.extend(current_loss_history)
