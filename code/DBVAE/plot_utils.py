"""
All functions used to save images.
"""
from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np

from torchvision.utils import save_image
from torchvision.utils import make_grid


def check_folder(name):
    """ Create folder if folder does not exist """
    Path(name).mkdir(parents=True, exist_ok=True)


def write_args_to_file(args, results_dir):
    """ Save all arguments in a file """
    with open('{}/arguments.txt'.format(results_dir), 'w') as file:
        for key, value in vars(args).items():
            file.write("{}: {}\n".format(key, str(value)))


def plot_images(images, result_path, folder, n_samples, epoch):
    """ Save n samples as one image """
    full_path = '{}/{}'.format(result_path, folder)
    check_folder(full_path)  # create destination folder

    save_image(make_grid(
        images[:n_samples, [2, 1, 0], :, :], nrow=int(math.sqrt(n_samples)),
        padding=2, normalize=True), '{}/{}.png'.format(full_path, epoch))


def plot_loss_curves(epochs, train_curve, val_curve, result_dir):
    """ Save a plot the loss of the training and validation set over epochs """
    full_path = '{}'.format(result_dir)
    check_folder(full_path)  # create destination folder


    # plot losses over epochs
    plt.plot(epochs, train_curve, label='train loss')
    plt.plot(epochs, val_curve, label='val loss')

    # set plot parameters
    plt.style.use('seaborn')
    plt.rcParams["figure.figsize"] = [10, 5]
    plt.legend(loc='lower right', bbox_to_anchor=(
        0., 1.02, 1., .102), ncol=2)
    plt.ylabel('loss')
    plt.xlabel('epochs')

    # save figure
    plt.savefig('{}/loss.png'.format(full_path))
    plt.clf()


def plot_probabilities_histogram(sample_probabilities, result_path, labels, epoch):
    """ Save a plot with the sample probability distribution """
    check_folder('{}/sample_histograms/'.format(result_path))

    # Show all sample probabilities in a histogram
    histogram_density, bin_edges = np.histogram(
        sample_probabilities[labels.squeeze() == 1], bins=10)

    # plot histogram
    plt.hist(bin_edges[:-1], bin_edges, weights=histogram_density, log=True)

    # set plot parameters
    plt.style.use('seaborn')
    plt.rcParams["figure.figsize"] = [10, 5]
    plt.ylabel('Counts')
    plt.xlabel('Edges')

    # save image
    plt.savefig('{}/sample_histograms/{}'.format(result_path, str(epoch)))
    plt.clf()


def plot_most_least_probable_samples(sample_probabilities, result_path, images, epoch):
    """ Save images of the most and least probable samples  """
    n_samples = 16

    # list with indices sorted from lowest to highest probabilities
    indices_by_prob = sample_probabilities.argsort()

    # get images with largest sampling probabilities
    biggest = indices_by_prob[-n_samples:]
    biggest_images = images[biggest].permute((0, 3, 1, 2))

    # get images with smallest sampling probabilities
    smallest = indices_by_prob[:n_samples]
    smallest_images = images[smallest].permute((0, 3, 1, 2))

    print("--- Probabilities: Min {}, Max {}, Difference: {}".format(
        sample_probabilities[smallest].mean(),
        sample_probabilities[biggest].mean(),
        sample_probabilities[biggest].mean() / \
        sample_probabilities[smallest].mean()
    ))

    # save images
    plot_images(biggest_images.float()/255, result_path,
                "/sample_probabilities/highest/", 16, epoch)
    plot_images(smallest_images.float()/255, result_path,
                "/sample_probabilities/lowest/", 16, epoch)


def save_probabilities(sample_probabilities, result_path, epoch):
    """ Save a csv with sampling probability of each sample """
    check_folder('{}/probabilities/'.format(result_path))
    with open('{}/probabilities/{}.csv'.format(
            result_path, epoch), 'a') as prob_file:
        for cell in np.nditer(sample_probabilities.squeeze()):
            prob_file.write(str(cell) + "\n")
