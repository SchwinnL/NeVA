import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import tikzplotlib
import os
import imageio
from matplotlib import cm


from src.CustomEnums import DataSetName
from src.data import get_stimuli_img, get_stimuli_img_sizes, get_dataset_information, is_stimuli_data
from src.SaveData import get_foveation_data_path
from src.foveation import get_foveation, calculate_blur, get_foveation_pos

def plot_image(images):
    plt.imshow(preprocess(images, True)[0])
    plt.show()

def plot_foveation_position(image, position):
    plt.imshow(image)
    plt.scatter([position[0]], [image.shape[0] - position[1]])
    plt.show()
    
def plot_from_foveation_position(conf, test_loader):
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

    plot_idxs = np.arange(0, len(test_loader.dataset.targets), test_loader.batch_size)

    if is_stimuli_data(dataset):
        img_sizes = get_stimuli_img_sizes(conf, dataset)
        img_paths = np.array(test_loader.dataset.imgs)
        img_names = [img[0].split("\\")[-1][:-4] for img in img_paths]
        if conf.plot_img_names is not None:
            plot_idxs = [i for i, name in enumerate(img_names) if name[:-1] in conf.plot_img_names]
        foveation_image_size = torch.tensor(img_sizes.reshape(img_sizes.shape[0], 2, 1, 1), dtype=torch.float32, device='cuda')

    suffix = "_adversarial" if conf.adversarial else ""

    forgetting_test = conf.forgetting_test
    foveation_data_path = get_foveation_data_path(conf, test_forgetting=forgetting_test, suffix=suffix) + "scanpath_data.pickle"
    predicted_classes_data_path = get_foveation_data_path(conf, test_forgetting=forgetting_test, suffix=suffix) + "label_dict.pickle"

    positions = pickle.load(open(foveation_data_path, 'rb'))
    positions = np.array(list(positions.values()))[:, :, :2]
    foveation_pos = get_foveation_pos(positions, foveation_image_size)
    predicted_classes = list(pickle.load(open(predicted_classes_data_path, 'rb')).values())
    predicted_classes = [classes.split(":")[1] for classes in predicted_classes]

    sigma = conf.blur_sigma if conf.blur_sigma is not None else 5
    forgetting = forgetting_test if forgetting_test is not None else conf.forgetting
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        B = X.shape[0]
        W = X.shape[2]
        min, max = i * test_loader.batch_size, (i + 1) * test_loader.batch_size
        # blur images
        blur = calculate_blur(X, conf.filter_size, conf.noise_factor, sigma=sigma)
        foveation_mask = torch.zeros((B, 1, image_size, image_size), device='cuda')
        pixel_positions = np.zeros((B, 7, 2))
        with torch.no_grad():
            for step in range(7):  # conf.deblur_steps
                # Generate mask from position #foveation_pos[min:max, step]
                current_foveation_mask = get_foveation(foveation_pos[min:max, step], conf.foveation_size, 1, image_size)
                idxs = current_foveation_mask.view(B, -1).argmax(1)
                xp = (idxs % W).cpu().numpy()
                yp = (idxs // W).cpu().numpy()
                current_pixel_positions = np.stack((xp[:, np.newaxis], yp[:, np.newaxis]), 1).squeeze()
                pixel_positions[:, step] = current_pixel_positions
                foveation_mask = foveation_mask * forgetting + current_foveation_mask
                current_blurring_mask = torch.ones((B, 1, image_size, image_size), device='cuda') - current_foveation_mask
                blurring_mask = torch.ones((B, 1, image_size, image_size), device='cuda') - foveation_mask
                blurring_mask = torch.clip(blurring_mask, 0, 1)
                for idx in plot_idxs:
                    if idx >= min and idx < max:
                        c_idx = idx - min
                        plot_foveation(conf, X, blur, blurring_mask, foveation_mask, f"{img_names[idx][:-1]}_{step}", ["PlotAfter" + suffix], idx=c_idx)
                        plot_pipeline(conf, X, blur, blurring_mask, foveation_mask, current_foveation_mask, current_blurring_mask, f"{img_names[idx][:-1]}_{step}", ["PlotAfter" + suffix], predicted_classes[idx], idx=c_idx)
                        plot_overlap(conf, X, foveation_mask, blur, blurring_mask, pixel_positions[:, :step + 1], f"{img_names[idx][:-1]}_{step}", ["PlotAfter" + suffix], predicted_classes[idx], idx=c_idx)
            if i < 5:
                plot_foveation_video(conf, img_names[idx][:-1], 10, ["PlotAfter" + suffix])


def plot_scanpath_diversity(conf):
    save_path = conf.image_save_path(sub_dir=["ScanpathDiversity"])

    suffix = "_adversarial_downstream" if conf.adversarial else ""
    data_path = conf.data_path + 'UVA/Scan_Paths/'
    df = pd.DataFrame()
    df_pl = pd.DataFrame()
    df_fix = pd.DataFrame()
    for ds in conf.test_dataset:

        for exp in conf.scan_path_experiments:
            positions, _ = load_scanpath_information(data_path + exp + "/" + ds.name + "/")
            path_length = np.sqrt(np.sum((np.roll(positions, 1, axis=1)[:, 1:] - positions[:, 1:]) ** 2, 2))
            average_path_length = np.sum(path_length, 1)
            quant = 5
            positions = (positions - positions.min()) / (positions.max() - positions.min())
            positions_quant = (quant * positions - 1e-4).astype(int)
            positions_quant = positions_quant[:, :, 0] + positions_quant[:, :, 1] * quant
            alphabet = np.array(list('abcdefghijklmnopqrstuvwxyz'))
            positions_letters = alphabet[positions_quant]

            unique_num = [len(np.unique(path)) for path in positions_letters]
            current_df = pd.DataFrame({"experiment": conf.scan_path_experiments[exp], "dataset": ds.name, "unique": unique_num, "path_length": average_path_length})
            path_length_df = pd.DataFrame({"experiment": conf.scan_path_experiments[exp], "dataset": ds.name, "path_length": path_length.flatten(), "fixations":np.arange(1, 10)[np.newaxis, :].repeat(len(path_length), 0).flatten()})
            df = pd.concat((df, current_df), ignore_index=True)
            df_pl = pd.concat((df_pl, path_length_df), ignore_index=True)

            sns.histplot(data=current_df, hue="experiment", x="path_length", bins=40)
            plt.xticks(rotation=90)
            plt.ylabel("Path length")
            save(save_path, "hist_" + conf.scan_path_experiments[exp] + "_" + ds.name)
            plt.clf()

        sns.histplot(data=df_pl[df_pl.dataset == ds.name], hue="experiment", x="path_length", bins=40)
        plt.xticks(rotation=90)
        plt.ylabel("Path length")
        save(save_path, "scanpath_length_histogram_" + ds.name)

        sns.boxplot(data=df_pl[df_pl.dataset == ds.name], hue="experiment", y="path_length", x="fixations")
        plt.xticks(rotation=90)
        plt.ylabel("Path length")
        save(save_path, "scanpath_length_fixation_" + ds.name)

    sns.boxplot(data=df, hue="experiment", y="path_length", x="dataset")
    plt.xticks(rotation=90)
    plt.ylabel("Path length")
    save(save_path, "scanpath_length_diversity")


    sns.boxplot(data=df, hue="experiment", y="unique", x="dataset")
    plt.xticks(rotation=90)
    plt.ylabel("\\#Diverse fixations")
    save(save_path, "scanpath_visited_diversity")


def plot_scanpath_diversity_grouped_by_labels(conf):
    save_path = conf.image_save_path(sub_dir=["ScanpathClassWise"])

    suffix = "_adversarial_downstream" if conf.adversarial else ""
    data_path = conf.data_path + 'UVA/Scan_Paths/'
    for exp in conf.scan_path_experiments:

        positions, label_dict = load_scanpath_information(data_path + exp + "/" + conf.test_dataset.name + "/")
        path_length = np.sum(np.sqrt(np.sum((np.roll(positions, 1, axis=1)[:, 1:] - positions[:, 1:]) ** 2, 2)), 1)
        quant = 5
        positions = (positions - positions.min()) / (positions.max() - positions.min())
        positions_quant = (quant * positions - 1e-4).astype(int)
        positions_quant = positions_quant[:, :, 0] + positions_quant[:, :, 1] * quant
        alphabet = np.array(list('abcdefghijklmnopqrstuvwxyz'))
        positions_letters = alphabet[positions_quant]

        unique_num = [len(np.unique(path)) for path in positions_letters]
        df = pd.DataFrame({"experiment":exp, "unique":unique_num, "label":label_dict[:, 1], "Path length":path_length})
        if conf.dataset == DataSetName.imagenet:
            sns.boxplot(data=df, x="label", y="unique")
            plt.xticks([], rotation=90)
            plt.xlabel("Class")
        else:
            sns.boxplot(data=df, x="label", y="unique")
            plt.xticks(rotation=90)
            plt.xlabel("Class")
        plt.ylabel("\\#Diverse Fixations")
        plt.ylim([0.5, 10.5])
        plt.yticks([2, 4, 6, 8, 10])
        save(save_path, exp)
        plt.clf()

        sns.boxplot(data=df, x="label", y="Path length")
        plt.xticks(rotation=90)
        plt.xlabel("Class")
        plt.ylabel("Path length")
        plt.ylim([0, 4500])
        save(save_path, exp + "_path_length")
        plt.clf()


def load_scanpath_information(path):
    file = open(path + "scanpath_data.pickle", 'rb')
    pickle_file = pickle.load(file)
    positions = np.array(list(pickle_file.values()))[:, :, :2]

    if os.path.isfile(path + "label_dict.pickle"):
        file = open(path + "label_dict.pickle", 'rb')
        pickle_file = pickle.load(file)
        label_dict = np.array(list(pickle_file.values()))
        label_dict = np.array([[tup.split(":")[0], tup.split(":")[1].title()] for tup in label_dict])
    else:
        label_dict = np.zeros((positions.shape[0], 2))
    return positions, label_dict

def plot_scanpath_loss(conf):
    save_path = conf.image_save_path(sub_dir=["Scanpath_Loss"])

    suffix = "_adversarial_downstream" if conf.adversarial else ""
    data_path = get_foveation_data_path(conf, None, suffix)

    file = open(data_path + "loss_history.pickle", 'rb')
    pickle_file = pickle.load(file)
    df = pd.DataFrame(pickle_file)
    for i, key in enumerate(pickle_file):
        df = df.rename(columns={key:i})
    sns.boxplot(data=df)
    plt.ylabel("Downstream loss")
    plt.xlabel("Fixations")
    save(save_path, conf.full_model_name())

def plot_overlap(conf, images, foveation, blur, blurring_mask, positions, name="", subdir=None, predicted_label = "", idx=0):
    dir = ["Overlap2"]
    if subdir is not None:
        dir += subdir

    positions = positions[idx]
    blurring_mask = preprocess(blurring_mask)[idx]
    images = preprocess(images)[idx]
    blur = preprocess(blur)[idx]
    foveation = preprocess(foveation)[idx]

    mask = np.repeat(foveation < 0.2, 3, axis=2)
    foveation_color = np.repeat(foveation, 3, axis=2)
    foveation_color[mask] = (images + (blur * blurring_mask))[mask]
    foveation_color[~mask] = plt.cm.viridis(foveation)[:, :, 0, :3][~mask]

    fig, axs = plt.subplots(3, 1)

    def plot_path(i):
        for step in range(steps):
            x = positions[step, 0]
            y = positions[step, 1]
            axs[i].scatter(x, y, c='red', s=1)
            if step > 0:
                arrow_s = 7
                arrow_w = 5
                d = 7
                x_prev = positions[step - 1, 0]
                dx = x - x_prev
                y_prev = positions[step - 1, 1]
                dy = y - y_prev

                dxn = dx / np.sqrt((dx ** 2 + dy ** 2))
                dyn = dy / np.sqrt((dx ** 2 + dy ** 2))

                xs = x_prev + d * dxn
                dxf = dx - 2 * d * dxn
                ys = y_prev + d * dyn
                dyf = dy - 2 * d * dyn

                axs[i].arrow(xs, ys, dxf, dyf, fc='k',
                             ec='k', lw=1, head_width=arrow_w, head_length=arrow_s, length_includes_head=True)

    axs[0].imshow(images)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    steps = positions.shape[0]

    axs[1].imshow(images + (blurring_mask * blur))
    axs[1].imshow(foveation_color, alpha=0.5)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    plot_path(1)

    axs[2].imshow(images + (blurring_mask * blur))
    axs[2].set_xticks([])
    axs[2].set_yticks([])

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.05)

    path = conf.image_save_path(dir) + name + predicted_label + "_Overlap"
    plt.savefig(path, bbox_inches='tight', dpi=800)
    plt.clf()
    plt.close(fig)

def plot_pipeline(conf, images, blur, mask, foveation, current_foveation, current_mask, name="", subdir=None, predicted_label = "", idx=0):
    dir = ["Pipeline"]
    if subdir is not None:
        dir += subdir

    images = preprocess(images)[idx]
    blur = preprocess(blur)[idx]
    mask = preprocess(mask)[idx]
    foveation = preprocess(foveation)[idx]
    current_foveation = preprocess(current_foveation)[idx]
    current_mask = preprocess(current_mask)[idx]

    path = conf.image_save_path(dir) + name[:-1] + predicted_label + "_Image"
    plt.imshow(images)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path, bbox_inches='tight')
    plt.clf()

    #path = conf.image_save_path(dir) + name + "_Image_Blured"
    #plt.imshow(images + blur)
    #plt.xticks([])
    #plt.yticks([])
    #plt.savefig(path)
    #plt.clf()

    path = conf.image_save_path(dir) + name + "_Image_Foveated"
    plt.imshow(images + (blur * mask))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path, bbox_inches='tight')
    plt.clf()

    path = conf.image_save_path(dir) + name + "_Image_Current_Foveated"
    plt.imshow(images + (blur * current_mask))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path, bbox_inches='tight')
    plt.clf()

    #path = conf.image_save_path(dir) + name + "_Blur_Mask"
    #plt.imshow(mask, vmin=0, vmax=1)
    #plt.xticks([])
    #plt.yticks([])
    #plt.savefig(path)
    #plt.clf()

    #path = conf.image_save_path(dir) + name[:-2] + "_Blur"
    #plt.imshow(blur, vmin=0, vmax=1)
    #plt.xticks([])
    #plt.yticks([])
    #plt.savefig(path)
    #plt.clf()

    path = conf.image_save_path(dir) + name + "_Current_Foveation"
    plt.imshow(current_foveation, vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path, bbox_inches='tight')
    plt.clf()

    path = conf.image_save_path(dir) + name + "_Foveation"
    plt.imshow(foveation, vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path, bbox_inches='tight')
    plt.clf()

def plot_foveation(conf, images, blur, mask, foveation, name="", subdir=None, reconstructed=None, idx=0):
    fig, axs = plt.subplots(2, 2 + (reconstructed is not None))

    images_ = preprocess(images, True)[idx]
    blur_ = preprocess(blur)[idx]
    mask_ = preprocess(mask)[idx]
    foveation_ = preprocess(foveation)[idx]
    if reconstructed is not None:
        reconstructed = preprocess(reconstructed)[idx]

    axs[0, 0].imshow(images_)
    axs[0, 0].set_title("Image")

    axs[0, 1].imshow(images_ + blur_)
    axs[0, 1].set_title("Blured")

    axs[1, 0].imshow(images_ + (blur_ * mask_))
    axs[1, 0].set_title("Foveated")

    axs[1, 1].imshow(foveation_, vmin=0, vmax=1)
    axs[1, 1].set_title("Attention")

    if reconstructed is not None:
        axs[0, 2].set_visible(False)
        axs[1, 2].imshow(reconstructed)
        axs[1, 2].set_title("Reconstructed")
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])

    for i in range(4):
        axs[i%2, i//2].set_xticks([])
        axs[i%2, i//2].set_yticks([])

    if name == "":
        plt.show()
    else:
        dir = ["Scan_Paths"]
        if subdir is not None:
            dir += subdir
        path = conf.image_save_path(dir) + name
        plt.savefig(path)

    plt.clf()
    plt.close(fig)

def plot_foveation_video(conf, file_name, steps, subdir):
    images = []
    dir = ["Scan_Paths"]
    if subdir is not None:
        dir += subdir
    path = conf.image_save_path(dir)
    filenames = [f"{file_name}_{i}" for i in range(steps)]
    for filename in filenames:
        if not os.path.isfile(path + filename + ".png"):
            return
        images.append(imageio.imread(path + filename + ".png"))
    imageio.mimsave(f'{path}{file_name}.gif', images)

def preprocess(x, clip=False):
    x_ = torch.permute(x, (0, 2, 3, 1)).detach().cpu().numpy()
    if clip:
        x_ = np.clip(x_, 0, 1)
    return x_

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def save(path, filename, bbox_inches_tight=True, set_axis_width=True, file_endings= [".png", ".tex"]):
    for file_ending in file_endings:
        filename_and_ending = filename + file_ending
        if "pdf" in file_ending:
            if not os.path.isdir(path + "PDF"):
                os.mkdir(path + "PDF")
            if bbox_inches_tight:
                plt.savefig(path + "PDF/" + filename_and_ending, bbox_inches='tight')
            else:
                plt.savefig(path + "PDF/" + filename_and_ending)
        elif "pgf" in file_ending:
            if not os.path.isdir(path + "PGF"):
               os.mkdir(path + "PGF")
            if bbox_inches_tight:
                plt.savefig(path + "PGF/" + filename_and_ending, bbox_inches='tight')
            else:
                plt.savefig(path + "PGF/" + filename_and_ending)
        elif "png" in file_ending:
            if not os.path.isdir(path + "PNG"):
               os.mkdir(path + "PNG")
            if bbox_inches_tight:
                plt.savefig(path + "PNG/" + filename_and_ending, bbox_inches='tight', dpi=900)
            else:
                plt.savefig(path + "PNG/" + filename_and_ending)
        elif ".tex" in file_ending:
            if not os.path.isdir(path + "Tikz"):
               os.mkdir(path + "Tikz")
            if set_axis_width:
                tikzplotlib.save(path + "Tikz/" + filename_and_ending, axis_width="\\linewidth", extra_tikzpicture_parameters=["trim axis left, trim axis right"], encoding='utf-8')
            else:
                tikzplotlib.save(path + "Tikz/" + filename_and_ending, extra_tikzpicture_parameters=["trim axis left, trim axis right"], encoding='utf-8')
    plt.clf()