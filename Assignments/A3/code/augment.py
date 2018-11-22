import os
import random
from scipy import ndarray
import skimage as sk
import skimage.io
from skimage import transform
from skimage import util


def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)


def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)


def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]


def transform(image_to_transform):
    # dictionary of the transformations functions we defined earlier
    available_transformations = {
        'rotate': random_rotation,
        'noise': random_noise,
        'horizontal_flip': horizontal_flip
    }

    # random num of transformations to apply
    num_transformations_to_apply = random.randint(1, len(available_transformations))

    num_transformations = 0
    transformed_image = image_to_transform
    while num_transformations <= num_transformations_to_apply:
        # choose a random transformation to apply for a single image
        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](transformed_image)
        num_transformations += 1
    return transformed_image


def process(image_path, new_image_path):
    image_to_transform = skimage.io.imread(image_path)
    transformed_image = transform(image_to_transform)
    skimage.io.imsave(new_image_path, transformed_image)


def main():
    original_folder_path = 'bird_dataset/train_images'
    new_folder_path = 'very_large_augmented_bird_dataset/train_images'
    minimum_n_examples = 5000

    for directory in os.listdir(original_folder_path):
        if not os.path.exists(os.path.join(new_folder_path, directory)):
            os.makedirs(os.path.join(new_folder_path, directory))

        image_filenames = os.listdir(os.path.join(original_folder_path, directory))
        sampled_image_filenames = random.choices(image_filenames, k=minimum_n_examples)

        for idx, image_filename in enumerate(sampled_image_filenames):
            new_image_filename = image_filename[:-4] + '_' + str(idx) + '.jpg'
            process(os.path.join(original_folder_path, directory, image_filename),
                    os.path.join(new_folder_path, directory, new_image_filename))


if __name__ == '__main__':
    main()