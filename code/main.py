"""
DB_VAE model (Amini and Soleimany, 2019), reproduced for FACT2020 course.
"""
import argparse
import copy

from DBVAE.functions import *
from DBVAE.db_vae import DBVAE
import DBVAE.data_utils as data_utils
import DBVAE.plot_utils as plot_utils

# -----------------------------------------------------------
# Select device and show information
# -----------------------------------------------------------

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('----------------------------------')
print('Using device for training:', DEVICE)
print('----------------------------------')
print()


def main():
    """
    Create a model, optimizer, and classification loss
    If data not present, download data and prepare training and validation set
    Initialize the Dataloaders
    For each epoch, create a new probabilistic dataloader,
    and run one training and validation epoch
    """

    # Create a model, optimizer, and classification loss
    model = DBVAE(z_dim=ARGS.z_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())

    # create a directory for results
    result_dir = "results"
    plot_utils.check_folder(result_dir)
    plot_utils.write_args_to_file(ARGS, result_dir)

    # create a file in which losses and accuracies are written during training
    with open('{}/loss_acc.csv'.format(result_dir), 'w') as losses_file:
        losses_file.write("train_loss,val_loss,train_acc,val_accc\n")

    # Download and prepare all datasets
    data_utils.prepare_datasets()


    # Load the dataset
    train_data, val_data = data_utils.load_datasets('data/{}'.format(ARGS.dataset))

    # Create dataloaders
    train_seq_dataloader = torch_data.DataLoader(train_data, batch_size=ARGS.batch_size)
    val_seq_dataloader = torch_data.DataLoader(val_data, batch_size=ARGS.batch_size)

    # lists to save training and validation losses
    train_curve, val_curve, val_accuracy_curve, epochs = [], [], [], []

    for c_epoch in range(ARGS.epochs):
        print("\n--- Starting epoch {} ---".format(c_epoch))

        if ARGS.disable_debias or c_epoch == 0:
            train_sampler = torch_data.RandomSampler(train_data)

        else:
            # create a weighted train_sampler
            train_sampler = get_train_sampler(
                train_seq_dataloader, model.encoder, train_data, ARGS, DEVICE,
                result_dir, c_epoch)

        # Initialize a dataloader with sampler with given probabilities
        train_prob_dataloader = torch_data.DataLoader(
            train_data, batch_size=ARGS.batch_size, sampler=train_sampler)

        # perform one training and validation epoch
        model.train()
        train_loss, train_accuracy = run_epoch(
            model, train_prob_dataloader, optimizer, c_epoch, result_dir, ARGS,
            DEVICE)

        model.eval()
        val_loss, val_accuracy = run_epoch(model, val_seq_dataloader, optimizer,
                                           c_epoch, result_dir, ARGS, DEVICE)

        with open('{}/loss_acc.csv'.format(result_dir), 'a') as losses_file:
            losses_file.write("{},{},{},{}\n".format(
                train_loss, val_loss, train_accuracy, val_accuracy))

        # save losses and val accuracy in lists
        train_curve.append(train_loss)
        val_curve.append(val_loss)
        val_accuracy_curve.append(val_accuracy)
        epochs.append(c_epoch)

        # save model if accuracy has increased
        if val_accuracy == max(val_accuracy_curve):
            print("--- New best validation accuracy ---")
            best_model = copy.deepcopy(model)
            best_epoch = c_epoch

    # After training, save the best model
    print("--- Training done. Best model after epoch {}".format(best_epoch))
    torch.save(best_model.state_dict(), '{}/model_epoch_{}.pth.tar'.format(result_dir, best_epoch))

    # save a picture of the loss curves
    plot_utils.plot_loss_curves(epochs, train_curve, val_curve, result_dir)

    # evaluate the model on the test set.
    print("--- Evaluating model on test set ---")
    evaluate_model(best_model, result_dir)


# -----------------------------------------------------------
# Compile ARGS and run main()
# -----------------------------------------------------------

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()

    # Arguments for training
    PARSER.add_argument('--dataset', type=str, default="faces",
                        help='The dataset which we try to train on')
    # PARSER.add_argument('--dataset', type=str, required=True,
    #                     help='The dataset which we try to train on')
    PARSER.add_argument('--epochs', default=60, type=int,
                        help='max number of epochs')
    PARSER.add_argument('--batch_size', default=24, type=int,
                        help='size of batch')

    # Arguments for debiasing
    PARSER.add_argument('--alpha', default=0.001, type=float,
                        help='debiasing rate')

    # Arguments for the VAE/model
    PARSER.add_argument('--z_dim', default=100, type=int,
                        help='dimensionality of the latent space')

    # Arguments for the loss function
    PARSER.add_argument('--loss_recon_reduce', default='sum', type=str,
                        help='Reduce the reconstruction loss: sum or mean')
    PARSER.add_argument('--loss_regularization_weight', default=1, type=float,
                        help='weight for the regularization loss function')
    PARSER.add_argument('--disable_debias', default=False, type=bool,
                        help='Disable the debias sampling')

    ARGS = PARSER.parse_args()

    main()
