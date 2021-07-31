"""
Functions used in main.py file.
"""
import csv
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np

import DBVAE.plot_utils as plot_utils
import DBVAE.debias_utils as debias_utils
import DBVAE.ppb_utils as ppb_utils


def compute_accuracy(y, y_pred):
    """ Returns classification accuracy of prediction """
    sigmoid = nn.Sigmoid()
    return (sigmoid(y_pred).round().squeeze() == y).float().mean()


def full_loss(x, y, mean, logvar, reconstruction, y_pred, ARGS, device):
    """
    Calculate the complete loss for the DB_VAE. Combination of reconstruction-,
    regularization- and classification loss.
    """
    reg_weight = ARGS.loss_regularization_weight
    reduce = ARGS.loss_recon_reduce

    # l2 loss as classification loss
    lx_loss = nn.MSELoss(reduction='none')

    # binary cross entropy as reconstruction loss
    criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)

    # calculate KL divergence as regularization loss
    reg_loss = -(1 / 2) * torch.sum(
        1 + logvar - mean.pow(2) - torch.exp(logvar), dim=1)

    # calculate and reduce reconstruction loss
    recon_loss_full = lx_loss(x, reconstruction) # Test mean
    if reduce == 'mean':
        recon_loss = torch.mean(recon_loss_full, dim=(1, 2, 3))
    else:
        recon_loss = torch.sum(recon_loss_full, dim=(1, 2, 3))

    classification_loss = criterion(y_pred.squeeze(), y)

    # weighted total loss
    loss = (reg_weight * reg_loss + recon_loss) * y + classification_loss

    return loss.mean()


def run_epoch(model, dataloader, optimizer, epoch, result_dir, ARGS, device):
    """
    Runs one training epoch.
    :param model: DB_VAE model to be trained.
    :param dataloader: Loads data batches, either sequential or probabilistic.
    :return: Epoch loss and accuracy
    """
    max_iters = int(len(dataloader.dataset) / ARGS.batch_size)

    total_epoch_loss = np.array([])
    temp_epoch_accuracy = 0
    samples = 0

    for iter, batch in zip(range(max_iters), dataloader):

        # prepare an x and y batch
        x, y = batch
        x = x.permute((0, 3, 1, 2)).to(device)
        y = y.float().to(device)

        # if model.training = True, calculate gradients
        with torch.set_grad_enabled(model.training):
            # do one forward pass
            mean, logvar, y_pred, reconstruction, z = model(x)

            loss = full_loss(x, y, mean, logvar, reconstruction, y_pred,
                             ARGS, device)

        accuracy = compute_accuracy(y, y_pred)

        # perform weight updates
        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # add batch loss and accuracy to arrays
        total_epoch_loss = np.append(total_epoch_loss, loss.item())
        temp_epoch_accuracy += accuracy.item()*x.shape[0]
        samples += x.shape[0]

        # get examples of original and reconstruction images if validating
        if not model.training:
            if iter == 0:
                plot_utils.plot_images(x, result_dir,
                                       'samples_rec/original/', 16, epoch)
                plot_utils.plot_images(reconstruction, result_dir,
                                       'samples_rec/reconstructed/', 16, epoch)

    average_epoch_loss = np.mean(total_epoch_loss)
    average_epoch_accuracy = temp_epoch_accuracy / samples

    if model.training:
        print("\n --- Training finished ---")
        print("Average epoch loss: {}, Average epoch accuracy: {}".format(
            average_epoch_loss, average_epoch_accuracy))
    else:
        print("\n --- Validation finished ---")
        print("Average epoch loss: {}, Average epoch accuracy: {}".format(
            average_epoch_loss, average_epoch_accuracy))

    return average_epoch_loss, average_epoch_accuracy


def get_train_sampler(train_seq_dataloader, encoder, train_data, ARGS, device, result_dir, epoch):
    """
    Calculate the probabilities for a sample to be sampled and create a sampler
    based on these probabilities.
    :param train_seq_dataloader: Loads complete dataset sequentially
    :param encoder: Encoding part of the DB_VAE
    :return: Weighted probabilities for data sampler
    """
    # get the latent mean of each latent variable for each sample
    latent_means = debias_utils.get_all_latent_means(
        train_seq_dataloader, encoder, ARGS.z_dim, device)

    # Calculate the probability of each sample based on the histograms
    sample_probabilities = \
        debias_utils.get_training_sample_probabilities(
            latent_means, train_data.labels, bins=10, alpha=ARGS.alpha)

    # save the sample probabilities in a csv file
    plot_utils.save_probabilities(sample_probabilities, result_dir, epoch)

    # plot figures with the most and least probable samples
    plot_utils.plot_most_least_probable_samples(
        sample_probabilities, result_dir, train_data.images, epoch)

    # plot a histogram with the amount of samples per probability bin
    plot_utils.plot_probabilities_histogram(sample_probabilities, result_dir,
                                            train_data.labels, epoch)

    # create the train sampler, used with the dataloader
    train_sampler = torch_data.WeightedRandomSampler(sample_probabilities,
                                                     len(sample_probabilities))

    return train_sampler


def evaluate_model(model, result_dir):
    """ Tests models classification accuracy faces on PPB dataset """
    test_accuracies = {}
    keys = ["male_lighter", "male_darker", "female_lighter", "female_darker"]

    # load the evaluator
    face_evaluator = ppb_utils.PPBFaceEvaluator()
    name = result_dir + "/test_accuracy.csv"

    total_test_accuracy = 0
    total_num_faces = 0

    # for each data group, calculate accuracies
    for key in keys:
        accuracy, num_faces = face_evaluator.evaluate([model], key)
        print(key, accuracy)

        test_accuracies[key] = accuracy
        total_test_accuracy += (num_faces * accuracy[0])
        total_num_faces += num_faces

    #calculate overall accuracy
    total_test_accuracy = total_test_accuracy / total_num_faces
    test_accuracies['total'] = total_test_accuracy

    # write accuracies to file
    w = csv.writer(open(name, "w"))
    for key, val in test_accuracies.items():
        w.writerow([key, val])
