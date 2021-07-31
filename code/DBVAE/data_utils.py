import requests
import os
import h5py
import torch
from torch.utils import data
from pathlib import Path
import tarfile
import sys
import numpy as np

def _write_to_h5(file, images, labels):
    print('- write to {}'.format(file))
    hf = h5py.File(file, 'w')
    hf.create_dataset('images', data=images)
    hf.create_dataset('labels', data=labels)
    hf.close()
    print(" DONE")


def prepare_datasets(big_dataset = True):
    Path('data').mkdir(parents=True, exist_ok=True)
    if big_dataset:
        _prepare_trainset()

    _prepare_testset()

def _prepare_testset():
    if os.path.isdir('data/PPB-2017'):
        print('- PPB dataset already downloaded and parsed :)')
        return
    print("---- PREPARE TESTSET ----")

    download_file('https://www.dropbox.com/s/l0lp6qxeplumouf/PPB.tar?dl=1', 'data/PPB.tar')
    print('Unpacking PPB file')
    import tarfile
    tf = tarfile.open('data/PPB.tar')
    tf.extractall('data')
    print("---- FINISHED TESTSET ----")
    return


def _prepare_trainset(name='faces', num_samples=-1, train_percentage=0.8):
    preprocess_dir = 'data/{}'.format(name)
    if os.path.isdir(preprocess_dir):
        print('- Train dataset already downloaded and parsed :)')
        return

    print("---- PREPARE TRAINSET ----")

    h5_file = 'data/faces.h5'
    download_file('https://www.dropbox.com/s/l5iqduhe0gwxumq/train_face.h5?dl=1', h5_file)

    print("- Opening file and reading labels/images")
    f = h5py.File(h5_file, 'r')
    print('- labels', end='')
    all_labels = f["labels"][:]
    print(" DONE")
    print('- images', end='')
    all_images = f["images"][:]
    print(" DONE")

    print('- Shuffle data')
    permutation = np.random.permutation(all_labels.shape[0])
    images = all_images[permutation]
    labels = all_labels[permutation]

    print('- Create test and training set')
    total_samples = num_samples if num_samples > 0 else all_labels.shape[0]
    images = images[:total_samples]
    labels = labels[:total_samples]

    num_train_samples = round(total_samples * train_percentage)
    train_data = (images[:num_train_samples], labels[:num_train_samples])
    test_data = (images[num_train_samples:], labels[num_train_samples:])

    print('- Write to files')

    if not os.path.exists(preprocess_dir):
        os.mkdir(preprocess_dir)

    _write_to_h5('{}/train.h5'.format(preprocess_dir), train_data[0], train_data[1])
    _write_to_h5('{}/test.h5'.format(preprocess_dir), test_data[0], test_data[1])

    print("---- FINISHED TRAINSET ----")
    return



def download_file(url, file, only_once=True):
    """ Download the dataset file and save it in a folder. """
    if only_once and os.path.isfile(file):
        print("Datafile already downloaded")
        return

    print("Start downloading datafile")

    r = requests.get(url, stream=True)

    # create directory
    folder = file.split("/")[0]
    Path(folder).mkdir(parents=True, exist_ok=True)

    # write the content to a file
    with open(file, 'wb') as f:
        for lines in r.iter_content(1024):
            f.write(lines)

    print("Finished downloading datafile")
    return


def load_datasets(path):
    """ Loads the training and validation data in classes. """
    if not os.path.isfile('{}/train.h5'.format(path)) or not \
            os.path.isfile('{}/test.h5'.format(path)):
        print("YOUR TRAIN/VALIDATION DATASET DOES NOT EXIST. CHOSE OR GENERATE ANOTHER DATASET.")
        sys.exit()

    # Load the datasets
    train_data = H5Dataset('{}/train.h5'.format(path))
    val_data = H5Dataset('{}/test.h5'.format(path))

    return train_data, val_data


class H5Dataset(data.Dataset):
    """
    The H5Dataset is a dataset which loads a H5 file with images and labels.
    """
    def __init__(self, file):
        self.file = file
        self.reader = h5py.File(self.file, 'r')

        self.images = []
        self.labels = []
        self.size = 0
        self._load_data()

    def _load_data(self):
        print("--- Start loading dataset ---")
        self.images = torch.from_numpy(self.reader['images'][:])
        self.labels = self.reader['labels'][:]

        self.size = self.reader['images'].shape[0]
        print("--- Finished loading dataset ---")

    def __getitem__(self, index):
        # normalize image values
        x = self.images[index].float()/255.0

        y = int(self.labels[index])

        return x, y

    def __len__(self):
        return self.reader['images'].shape[0]
