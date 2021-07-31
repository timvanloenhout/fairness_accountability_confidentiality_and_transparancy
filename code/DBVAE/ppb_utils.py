"""
Class to make use of the PPB test set. Adapted from the notebook by A. Amini
for the MIT Deep Learning course.
"""
import os
import torch
import cv2
import numpy as np

IM_SHAPE = (64, 64, 3)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PPBFaceEvaluator:
    """ Evaluate algorithm on PPB dataset. """
    def __init__(self, skip=1):
        # create dictionary with image name as key and gender and skin as tuple
        self.meta_data = {}
        with open('data/PPB-2017/PPB-2017-metadata.csv') as f:
            for line in f.read().split('\n'):
                _, name, gender, _, skin, _ = line.split(',')
                self.meta_data[name] = (gender.lower(), skin.lower())

        # get every 'skip'th image
        image_files = sorted(os.listdir('data/PPB-2017/imgs'))[::skip]

        # create a dictionary with gender_skin as key, and list of
        # corresponding images as value
        self.raw_images = {
            'male_darker': [],
            'male_lighter': [],
            'female_darker': [],
            'female_lighter': [],
        }

        # fill raw_images dictionary
        for filename in image_files:
            if not filename.endswith(".jpg"):
                continue
            image = cv2.imread(os.path.join(
                "data/PPB-2017/imgs", filename))[:, :, ::-1]
            gender, skin = self.meta_data[filename]
            self.raw_images[gender + '_' + skin].append(image)


    def evaluate(self, models_to_test, key, patch_stride=0.2, patch_depth=5):
        """
        Creates patches of different sizes relative to the original image size,
        and forwards them all through the model. If one of the patches is
        classified as face, than the image itself is labeled as face.
        :param models_to_test: List with models that needs to be tested.
        :param key: Data group that you want the accuracy of.
        :return: Accuracy
        """
        correct_predictions = [0.0]*len(models_to_test)

        num_faces = len(self.raw_images[key])
        print(key, "amount of faces:", num_faces)

        for face_idx in range(num_faces):
            # load image
            image = self.raw_images[key][face_idx]
            _, width, _ = image.shape

            # create patches of size [64, 64] from image
            patches, _ = slide_square(
                image, patch_stride, width/2, width, patch_depth)
            patches = np.stack(patches, axis=0)
            patches = torch.tensor(patches).float() / 255.0
            patches = patches.permute(0, 3, 1, 2).to(DEVICE)
            patches = patches[:, [2, 1, 0], :, :]

            for model_idx, model in enumerate(models_to_test):
                # perform a forward pass
                _, _, out, _, _ = model(patches)

                # get highest probability
                y = out.detach().cpu().numpy()
                y_inds = np.argsort(y.flatten())
                most_likely_prob = y[y_inds[-1]]

                # add one to correct predictions if prob > 0
                if most_likely_prob >= 0.0:
                    correct_predictions[model_idx] += 1

        # calculate accuracy
        accuracy = [correct_predictions[i]/num_faces
                    for i, _ in enumerate(models_to_test)]

        return accuracy, num_faces


def slide_square(img, stride, min_size, max_size, n):
    """
    Function to slide a square across image and extract square regions.
    :param img: The image
    :param stride: (0,1], provides the fraction of the dimension for which to
    slide to generate a crop
    :param min_size: Minimum square size.
    :param max_size: Maximum square size.
    :param n: Number of different sizes including min_size, max_size
    :return: Patches and top left and botton right corner locations of patches
    """
    img_h, img_w = img.shape[:2]

    # get square sizes
    square_sizes = np.linspace(min_size, max_size, n, dtype=np.int32)
    square_images = []
    # list of list of tuples: [(i1,j1), (i2,j2)] where i1,j1 is
    # the top left corner; i2,j2 is bottom right corner
    square_bbox = []

    # for each of the square_sizes
    for sq_dim in square_sizes:

        stride_length = int(stride*sq_dim)

        # set top left corners from which you want to take a square
        stride_start_i = range(0, int(img_h-sq_dim+1), stride_length)
        stride_start_j = range(0, int(img_w-sq_dim+1), stride_length)

        # for every corner
        for i in stride_start_i:
            for j in stride_start_j:

                # set top left and bottom right corner
                square_top_left = (i, j)
                square_bottom_right = (i+sq_dim, j+sq_dim)
                square_corners = (square_top_left, square_bottom_right)

                # get images square
                square_image = img[i:i+sq_dim, j:j+sq_dim]

                # resize to DB_VAE imput size
                square_resize = cv2.resize(square_image, IM_SHAPE[:2],
                                           interpolation=cv2.INTER_NEAREST)

                # append to list of images and bounding boxes
                square_images.append(square_resize)
                square_bbox.append(square_corners)

    return square_images, square_bbox
